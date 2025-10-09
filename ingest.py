# ingest.py
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

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
    print(f"📄 Ingesting {pdf_path} with unstructured...")

    # Use unstructured to partition the PDF into smart, structural elements
    elements = partition_pdf(pdf_path, strategy="hi_res")

    docs, metas, ids = [], [], []

    for element in elements:
        chunk = str(element)

        # --- FIX: Only add the chunk if it's not empty or just whitespace ---
        if chunk and chunk.strip():
            # Create richer metadata using the element's category
            metadata = {
                "source": os.path.basename(pdf_path),
                "element_type": element.category,
                "text": chunk
            }
            if hasattr(element.metadata, 'page_number'):
                metadata['page'] = element.metadata.page_number

            docs.append(chunk)
            metas.append(metadata)
            ids.append(str(uuid.uuid4()))

    if not docs:
        print(f"No text extracted from {pdf_path}. Skipping.")
        return

    # The rest of the function remains the same
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


