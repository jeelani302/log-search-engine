from pydantic import BaseModel, Field
from typing import List, Optional

class IngestRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Path to the PDF or TXT file to ingest.",
        examples=["data/sample_sop.txt"]
    )

class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    collection_name: str

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="User's plain-English question.",
        examples=["What should we do if a shipment is stuck at customs?"]
    )
    top_k: int = Field(
        3,
        description="How many chunks to retrieve."
    )

class AnswerResponse(BaseModel):
    answer: str = Field(description="The LLM's generated answer grounded in the retrieved context.")
    sources: List[str] = Field(description="The source text chunks used as evidence.")
