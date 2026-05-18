from langchain_chroma import Chroma
import os
from app.ingestion import embeddings, CHROMA_DIR

def retrieve_context(question: str, top_k: int = 3) -> list[str]:
    """
    Embeds the user's question, searches ChromaDB for the closest matching
    chunks, and returns the raw text of those chunks.
    """
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            "ChromaDB not found. Please ingest a document first "
            "using the /ingest endpoint."
        )

    # Connect to the existing Chroma database on disk
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Search the vector database for the top_k most similar chunks
    results = vector_store.similarity_search(question, k=top_k)

    # Extract just the raw text strings to pass to the LLM
    context_chunks = [doc.page_content for doc in results]
    
    return context_chunks
