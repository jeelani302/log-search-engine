"""
llm_client.py — Supports Gemini (cloud) and Ollama (local) for answering RAG queries.
"""

import os
from dotenv import load_dotenv
import ollama as ollama_sdk
from google import genai
from google.genai import types

from app.prompts import build_rag_prompt

load_dotenv()

def _ask_gemini(prompt: str) -> str:
    """Send the RAG prompt to Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, # Low temp for factual answers
            ),
        )
    except Exception as exc:
        raise Exception(f"Gemini API call failed: {exc}") from exc

    return response.text.strip()

def _ask_ollama(prompt: str) -> str:
    """Send the RAG prompt to local Ollama."""
    model = os.getenv("OLLAMA_MODEL", "phi4")
    try:
        response = ollama_sdk.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
    except Exception as exc:
        raise Exception(f"Ollama API call failed: {exc}\nEnsure Ollama is running.") from exc

    return response["message"]["content"].strip()


def generate_answer(context_chunks: list[str], question: str) -> str:
    """
    Given a list of retrieved text chunks and a user question,
    build the prompt and query the LLM to get an answer.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    prompt = build_rag_prompt(context_chunks, question)

    if provider == "ollama":
        return _ask_ollama(prompt)
    else:
        return _ask_gemini(prompt)
