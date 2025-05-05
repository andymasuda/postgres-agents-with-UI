import psycopg2
import csv
import json
import os
import sys
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from docx import Document
from io import BytesIO
import fitz

# Load environment variables
load_dotenv("../.env")

# Increase the maximum field size limit
csv.field_size_limit(sys.maxsize)

# Fetch the connection string from the environment variable
CONN_STR = os.getenv("AZURE_PG_CONNECTION")

# Fetch the OpenAI settings from environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Fetch the Embedding model name from environment variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(CONN_STR)

# Create a cursor object using the connection
cur = conn.cursor()

# Enable the required extensions
def create_extensions(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS azure_ai")
    print("Extensions created successfully")
    conn.commit()

# Setup OpenAI
def create_openai_connection(cur):
    cur.execute(f"SELECT azure_ai.set_setting('azure_openai.endpoint', '{AZURE_OPENAI_ENDPOINT}')")
    cur.execute(f"SELECT azure_ai.set_setting('azure_openai.subscription_key', '{AZURE_OPENAI_API_KEY}')")
    print("OpenAI connection established successfully")
    conn.commit()

# Create the document_chunks table
def create_tables(cur):
    cur.execute("DROP TABLE IF EXISTS document_chunks")
    conn.commit()

    cur.execute("""
        CREATE TABLE document_chunks (
            id SERIAL PRIMARY KEY,
            chunk TEXT,
            embeddings FLOAT8[]
        )
    """)
    print("Document chunks table created successfully")
    conn.commit()

# Encapsulate the query logic into a function
def create_tsvector_column_and_index(cur):
    """
    Adds a tsvector column for full-text search, creates a GIN index on it,
    and sets up a trigger to maintain the column.
    """
    # Add a tsvector column for full-text search
    cur.execute("""
        ALTER TABLE document_chunks
        ADD COLUMN IF NOT EXISTS tsvector_column tsvector
    """)
    
    # Populate the tsvector column with initial data
    cur.execute("""
        UPDATE document_chunks
        SET tsvector_column = to_tsvector('english', chunk)
    """)
    
    # Create a GIN index on the tsvector column
    cur.execute("""
        CREATE INDEX IF NOT EXISTS gin_index_tsvector
        ON document_chunks USING gin(tsvector_column)
    """)
    
    # Create a trigger to automatically update the tsvector column
    cur.execute("""
        CREATE OR REPLACE FUNCTION update_tsvector_column()
        RETURNS TRIGGER AS $$
        BEGIN
          NEW.tsvector_column := to_tsvector('english', NEW.chunk);
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS tsvector_update_trigger
        BEFORE INSERT OR UPDATE ON document_chunks
        FOR EACH ROW
        EXECUTE FUNCTION update_tsvector_column()
    """)
    
    print("tsvector column, GIN index, and trigger created successfully")
    conn.commit()

# Retrieve document from Azure Storage, split into chunks, and insert embeddings
def ingest_data_and_add_embeddings(cur, blob_name, container_name):
    # Initialize Azure Blob Storage client
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download the blob content as bytes
    blob_data = blob_client.download_blob().readall()

    # Determine the file type and read the document
    if blob_name.lower().endswith('.docx'):
        doc = Document(BytesIO(blob_data))
        document = "\n".join([para.text for para in doc.paragraphs])
    elif blob_name.lower().endswith('.pdf'):
        pdf_document = fitz.open(stream=blob_data, filetype="pdf")
        document = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            document += page.get_text()
    else:
        raise ValueError("Unsupported file type")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(document)

    # Initialize Azure OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version="2024-07-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )

    # Generate embeddings and insert into the database
    print("Generating embeddings and inserting into the database...")
    for chunk in chunks:
        response = openai_client.embeddings.create(input=[chunk], model=os.getenv('EMBEDDING_MODEL_NAME'))
        embedding = response.data[0].embedding
        cur.execute("INSERT INTO document_chunks (chunk, embeddings) VALUES (%s, %s)", (chunk, embedding))
    conn.commit()
    print("Document chunks and embeddings inserted successfully")

#create_extensions(cur)
#create_openai_connection(cur)
#create_tables(cur)
create_tsvector_column_and_index(cur)
#ingest_data_and_add_embeddings(cur, os.getenv('AZURE_BLOB_NAME'), os.getenv('AZURE_BLOB_CONTAINER_NAME'))

# Close the cursor and connection
cur.close()
conn.close()

print("All Data loaded successfully!")