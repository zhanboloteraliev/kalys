# ingest.py
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter # --- NEW ---

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

OPENAI_MODEL = "text-embedding-3-small"
PDF_DIR = "data/pdfs"

# -----------------------------
# Initialize OpenAI client
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Initialize Pinecone client
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
    print(f"Index '{INDEX_NAME}' already exists. Deleting it to ensure a clean re-ingestion.")
    pc.delete_index(name=INDEX_NAME)

print(f"Creating new index '{INDEX_NAME}'...")
pc.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
)

index = pc.Index(INDEX_NAME)

# -----------------------------
# --- NEW --- Initialize the smart text splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""] # Defines the order of splitting priority
)

# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages

# -----------------------------
# --- REMOVED --- The old, naive chunk_text function is no longer needed.
# -----------------------------

def embed_texts(texts, batch_size=50):
    """Call OpenAI embeddings API in batches"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai_client.embeddings.create(
            model=OPENAI_MODEL,
            input=batch
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings

def ingest_pdf(pdf_path):
    print(f"📄 Ingesting {pdf_path} ...")
    pages = extract_text_from_pdf(pdf_path)
    docs, metas, ids = [], [], []

    for page in pages:
        # --- MODIFIED --- Use the new text_splitter object here
        page_chunks = text_splitter.split_text(page["text"])
        for i, chunk in enumerate(page_chunks):
            docs.append(chunk)
            metas.append({
                "source": os.path.basename(pdf_path),
                "page": page["page"],
                "chunk_index": i,
                "text": chunk
            })
            ids.append(str(uuid.uuid4()))

    if not docs:
        print(f"No text extracted from {pdf_path}. Skipping.")
        return

    embeddings = embed_texts(docs)
    vectors = [(id_, vec, meta) for id_, vec, meta in zip(ids, embeddings, metas)]

    BATCH_SIZE = 50
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
    print(f"   Successfully ingested {len(vectors)} vectors for {os.path.basename(pdf_path)}.")

# -----------------------------
# Ingest all PDFs in folder
# -----------------------------
for f in os.listdir(PDF_DIR):
    if f.lower().endswith(".pdf"):
        ingest_pdf(os.path.join(PDF_DIR, f))

print("\n✅ All PDFs ingested into Pinecone!")


