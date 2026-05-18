"""
main.py — FastAPI application entry point for the Smart Logistics RAG System.

Run with:
    uvicorn app.main:app --reload

Then visit http://localhost:8000/docs for the interactive API explorer.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import IngestRequest, IngestResponse, QueryRequest, AnswerResponse
from app.ingestion import ingest_document
from app.retriever import retrieve_context
from app.llm_client import generate_answer

app = FastAPI(
    title="Smart Logistics RAG Search Engine",
    description="A Retrieval-Augmented Generation (RAG) system for ClickPost logistics SOPs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def health_check():
    """Quick check that the server is running."""
    return {"status": "ok", "message": "Logistics RAG Agent is live 🚀"}

@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["RAG Pipeline"],
    summary="Ingest a document into the knowledge base"
)
def ingest_endpoint(request: IngestRequest):
    """
    **Upload a logistics SOP to the knowledge base.**
    
    The document will be chunked and converted into vector embeddings
    stored in a local ChromaDB instance.
    """
    try:
        result = ingest_document(request.file_path)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

@app.post(
    "/query",
    response_model=AnswerResponse,
    tags=["RAG Pipeline"],
    summary="Ask a question about the logistics SOPs"
)
def query_endpoint(request: QueryRequest):
    """
    **Ask a plain-English question.**
    
    The system will:
    1. Search ChromaDB for the most relevant document chunks.
    2. Pass those chunks to the LLM (Gemini or Ollama).
    3. Return a grounded, accurate answer with no hallucination.
    """
    try:
        # Step 1: Retrieve relevant chunks
        context_chunks = retrieve_context(request.question, request.top_k)
        
        if not context_chunks:
            return AnswerResponse(
                answer="No relevant information found in the knowledge base.",
                sources=[]
            )

        # Step 2: Generate answer using the LLM
        answer = generate_answer(context_chunks, request.question)
        
        # Step 3: Return answer and sources
        return AnswerResponse(
            answer=answer,
            sources=context_chunks
        )
    except RuntimeError as exc: # e.g., ChromaDB not initialized
        raise HTTPException(status_code=400, detail=str(exc))
    except EnvironmentError as exc: # e.g., missing API key
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
