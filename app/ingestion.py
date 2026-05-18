import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_store"

# Initialize local embedding model globally to avoid reloading on every request
# all-MiniLM-L6-v2 is small, fast, and completely free (runs locally)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_document(file_path: str) -> dict:
    """
    Loads a text or PDF file, splits it into chunks, and stores them in ChromaDB.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load document
    if file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .txt or .pdf file.")
    
    documents = loader.load()

    # 2. Split into chunks
    # ~500 chars is good for capturing a single rule/procedure.
    # 50 chars overlap ensures we don't cut a sentence in half and lose context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Store in ChromaDB
    # We clear the existing database first to prevent duplicate chunks 
    # if you hit the ingest endpoint multiple times.
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    vector_store.delete_collection()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return {
        "status": "success",
        "chunks_stored": len(chunks),
        "collection_name": vector_store._collection.name
    }
