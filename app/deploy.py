"""Lightweight, stateless deployment entry point for free hosting."""

from __future__ import annotations

import math
import os
import re
import secrets
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _chunks_from_text(text: str, document_name: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    sections = re.split(r"\n(?=##?\s)", text)
    for index, section in enumerate(sections, start=1):
        content = section.strip()
        if not content:
            continue
        heading = next(
            (line.lstrip("# ") for line in content.splitlines() if line.startswith("#")),
            f"Section {index}",
        )
        chunks.append({"document": document_name, "section": heading, "content": content})
    return chunks


def _bundled_chunks() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        chunks.extend(_chunks_from_text(path.read_text(encoding="utf-8"), path.name))
    return chunks


DOCUMENTS = _bundled_chunks()
KNOWLEDGE_BASES: dict[str, list[dict[str, str]]] = {}


def retrieve(
    question: str, documents: list[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    query = Counter(_tokens(question))
    if not query:
        return []

    document_frequency = Counter(
        token for document in documents for token in set(_tokens(document["content"]))
    )
    scored: list[tuple[float, dict[str, str]]] = []
    total = max(len(documents), 1)
    for document in documents:
        terms = Counter(_tokens(document["content"]))
        score = 0.0
        for token, query_count in query.items():
            if token not in terms:
                continue
            inverse_frequency = math.log((total + 1) / (document_frequency[token] + 1)) + 1
            score += query_count * (1 + math.log(terms[token])) * inverse_frequency
        if score:
            scored.append((score, document))

    return [document for _, document in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def _extractive_answer(context: list[str]) -> str:
    if not context:
        return "I don't have information about that in the provided documents."
    lines = [line.strip(" -*") for line in context[0].splitlines()]
    useful = [line for line in lines if line and not line.startswith("#")]
    return " ".join(useful[:5])


def answer(question: str, context: list[str]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _extractive_answer(context)

    from google import genai

    prompt = (
        "Answer using only the context. If it is absent, say you do not have "
        "that information.\n\nContext:\n"
        + "\n\n---\n\n".join(context)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        contents=prompt,
    )
    return (response.text or "").strip()


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)
    knowledge_base_id: str | None = None


class IngestTextRequest(BaseModel):
    name: str = Field(default="uploaded-sop.txt", min_length=1, max_length=120)
    content: str = Field(min_length=20, max_length=200_000)


app = FastAPI(title="Smart Logistics Search Engine", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(DOCUMENTS)}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    documents = DOCUMENTS
    if request.knowledge_base_id:
        documents = KNOWLEDGE_BASES.get(request.knowledge_base_id, [])
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="This uploaded SOP session expired. Please add the SOP again.",
            )
    sources = retrieve(request.question, documents, request.top_k)
    context = [source["content"] for source in sources]
    try:
        generated = answer(request.question, context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc
    return {"answer": generated, "sources": sources}


@app.post("/ingest-text")
def ingest_text(request: IngestTextRequest) -> dict:
    safe_name = Path(request.name).name
    chunks = _chunks_from_text(request.content, safe_name)
    if not chunks:
        raise HTTPException(status_code=400, detail="No searchable text was found.")
    knowledge_base_id = secrets.token_urlsafe(18)
    if len(KNOWLEDGE_BASES) >= 20:
        KNOWLEDGE_BASES.pop(next(iter(KNOWLEDGE_BASES)))
    KNOWLEDGE_BASES[knowledge_base_id] = chunks
    return {
        "status": "ready",
        "knowledge_base_id": knowledge_base_id,
        "document": safe_name,
        "sections": len(chunks),
    }


@app.get("/demo")
def demo() -> dict:
    return {
        "knowledge_base": {
            "name": "ClickPost Logistics Operations — Standard Operating Procedures",
            "file": "data/sample_sop.txt",
            "version": "1.4",
            "updated": "2026-05-15",
            "sections": [source["section"] for source in DOCUMENTS if source["section"].startswith(tuple("1234"))],
        },
        "suggested_questions": [
            "What should we do when a shipment is stuck at customs for more than 48 hours?",
            "What is the response procedure for carrier webhook failures?",
            "When should a shipment be marked as RTO Initiated?",
            "What should a hub do with a visibly damaged package?",
            "When is a missing package marked as Lost in Transit?",
        ],
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "deploy.html")
