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

def create_tables(cur):
    cur.execute("DROP TABLE IF EXISTS invoices")
    conn.commit()

    cur.execute("""
        CREATE TABLE invoices (
            id SERIAL PRIMARY KEY,
            "ID" INT,
            "FiscalWeekBeginDate" TEXT,
            "Invoice Date" TEXT,
            "Region" TEXT,
            "Facility Name" TEXT,
            "Branch Id" TEXT,
            "Channel" TEXT,
            "soldto_name" TEXT,
            "shipto_name" TEXT,
            "Product Type" TEXT,
            "Major Code" TEXT,
            "Major Desc" TEXT,
            "Mid Code" TEXT,
            "Mid Desc" TEXT,
            "Minor Code" TEXT,
            "Minor Desc" TEXT,
            "Item" TEXT,
            "Item Desc" TEXT,
            "Sales" FLOAT8,
            "Gross Profit" FLOAT8,
            "GM Percent" FLOAT8,
            "TLE" FLOAT8,
            embeddings FLOAT8[]
        )
    """)
    print("Invoice table created successfully")
    conn.commit()

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def ingest_csv_and_add_embeddings(cur, csv_path):
    # Initialize Azure OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version="2024-07-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        print("Generating embeddings and inserting into the database from CSV...")
        for row in reader:
            # Concatenate all fields into a single string for embedding
            chunk = " | ".join([str(value) for value in row.values()])
            response = openai_client.embeddings.create(input=[chunk], model=os.getenv('EMBEDDING_MODEL_NAME'))
            embedding = response.data[0].embedding

            # Insert each field and the embedding into the invoice table
            cur.execute("""
                INSERT INTO invoices (
                    "ID", "FiscalWeekBeginDate", "Invoice Date", "Region", "Facility Name", "Branch Id", "Channel",
                    "soldto_name", "shipto_name", "Product Type", "Major Code", "Major Desc", "Mid Code", "Mid Desc",
                    "Minor Code", "Minor Desc", "Item", "Item Desc", "Sales", "Gross Profit", "GM Percent", "TLE", embeddings
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                int(row["ID"]),
                row["FiscalWeekBeginDate"],
                row["Invoice Date"],
                row["Region"],
                row["Facility Name"],
                row["Branch Id"],
                row["Channel"],
                row["soldto_name"],
                row["shipto_name"],
                row["Product Type"],
                row["Major Code"],
                row["Major Desc"],
                row["Mid Code"],
                row["Mid Desc"],
                row["Minor Code"],
                row["Minor Desc"],
                row["Item"],
                row["Item Desc"],
                safe_float(row["Sales"]),
                safe_float(row["Gross Profit"]),
                safe_float(row["GM Percent"]),
                safe_float(row["TLE"]),
                embedding
            ))
        conn.commit()
        print("CSV rows and embeddings inserted successfully")

def setup_fts_columns(cur):
    """
    Adds and populates the tsvector column for full-text search and creates a GIN index.
    """
    # Add the tsvector column if it doesn't exist
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='invoices' AND column_name='tsv'
            ) THEN
                ALTER TABLE invoices ADD COLUMN tsv tsvector;
            END IF;
        END
        $$;
    """)
    conn.commit()

    # Populate the tsvector column with concatenated text fields
    cur.execute("""
        UPDATE invoices
        SET tsv = to_tsvector('english',
            coalesce("soldto_name", '') || ' ' || coalesce("shipto_name", '') || ' ' ||
            coalesce("Major Desc", '') || ' ' || coalesce("Mid Desc", '') || ' ' ||
            coalesce("Minor Desc", '') || ' ' || coalesce("Item Desc", '')
        );
    """)
    conn.commit()

    # Create a GIN index on the tsvector column if it doesn't exist
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE tablename='invoices' AND indexname='invoices_tsv_idx'
            ) THEN
                CREATE INDEX invoices_tsv_idx ON invoices USING GIN(tsv);
            END IF;
        END
        $$;
    """)
    conn.commit()
    print("FTS tsvector column and GIN index set up successfully")

# Example usage:
#create_extensions(cur)
#create_openai_connection(cur)
#create_tables(cur)
#ingest_csv_and_add_embeddings(cur, "bluelinxsmalldata.csv")
setup_fts_columns(cur)

# Close the cursor and connection
cur.close()
conn.close()

print("All Data loaded successfully!")