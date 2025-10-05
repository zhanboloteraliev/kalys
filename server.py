import os
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import OpenAI, APIError
from pinecone import Pinecone, PineconeException
from dotenv import load_dotenv

load_dotenv()
# === Load API keys from env ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("kalysbot")

# FastAPI app
origins = [
    "https://www.kalysbot.com",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:63342",
    "https://kalys-fzbatnr2y-zhans-projects-e9448977.vercel.app"
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Retrieval function ===
def retrieve_context(query, top_k=4, selected_files=None):
    # Embed query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    if selected_files:
        pinecone_filter = {"source": {"$in": selected_files}}
        results = index.query(vector=query_embedding, top_k=top_k, filter=pinecone_filter, include_metadata=True)
    else:
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    sources = []
    snippets = []
    for match in results.matches:
        metadata = match.metadata
        sources.append({
            "uri": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "?"),
            "snippet": metadata.get("text", "")[:500]
        })
        snippets.append(metadata.get("text", ""))

    return snippets, sources

PDF_DIR = "data/pdfs" # Make sure this is defined at the top level

# Add this new endpoint to your server
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/pdf', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found.")

# === Chat completion with RAG context ===
@app.post("/rag")
async def rag(request: Request):
    data = await request.json()
    question = data.get("question", "")
    history = data.get("history", [])
    top_k = int(data.get("top_k", 4))
    selected_files = data.get("filters", [])

    # --- Error handling for Pinecone ---
    try:
        snippets, sources = retrieve_context(question, top_k, selected_files)
    except PineconeException as e:
        print(f"Pinecone error: {e}")
        # Return a specific error message to the frontend
        raise HTTPException(status_code=503, detail="Error communicating with the document database.")
    except Exception as e:
        print(f"An unexpected error occurred during retrieval: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


    # Build messages for GPT
    messages = [
        {"role": "system", "content": "You are a legal RAG assistant for Kyrgyz law..."}
    ]
    for msg in history:
        if "content" in msg and msg["content"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    context_text = "\n\n".join(snippets)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"})

    # --- Error handling for OpenAI ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = response.choices[0].message.content
    except APIError as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(status_code=503, detail="Error communicating with the language model.")
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
