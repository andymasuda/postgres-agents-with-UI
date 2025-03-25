import os
import fitz
import logging
import psycopg2
from io import BytesIO
from docx import Document
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Retrieve embeddings from Azure OpenAI
def get_embeddings(text):
    model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')
    response = openai_client.embeddings.create(input=[text], model=model)
    embedding_data = response.data[0].embedding
    return embedding_data

# Read DOCX
def read_docx(blob_data):
    doc = Document(BytesIO(blob_data))
    return "\n".join([para.text for para in doc.paragraphs])

# Read PDF
def read_pdf(blob_data):
    pdf_document = fitz.open(stream=blob_data, filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize Azure OpenAI client for embeddings
openai_client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
)

# Load and split document
blob_name = os.getenv('AZURE_BLOB_NAME')
blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
blob_client = blob_service_client.get_blob_client(container=os.getenv('AZURE_BLOB_CONTAINER_NAME'), blob=blob_name)

# Download the blob content as bytes
blob_data = blob_client.download_blob().readall()

# Determine the file type and read the document
if blob_name.lower().endswith('.docx'):
    document = read_docx(blob_data)
elif blob_name.lower().endswith('.pdf'):
    document = read_pdf(blob_data)
else:
    raise ValueError("Unsupported file type")

# Use LangChain for chunk splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(document)

# Insert document embeddings into PostgreSQL
conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST'),
    database=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD')
)

cur = conn.cursor()

# Create table if it does not exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS document_embeddings (
        id SERIAL PRIMARY KEY,
        document_chunk TEXT,
        embedding FLOAT8[]
    )
""")

for chunk in chunks:
    embedding = get_embeddings(chunk)
    cur.execute("INSERT INTO document_embeddings (document_chunk, embedding) VALUES (%s, %s)", (chunk, embedding))

conn.commit()
cur.close()
conn.close()

logger.info("Document embeddings inserted into PostgreSQL successfully.")