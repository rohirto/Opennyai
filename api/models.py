# api/models.py

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ======================================================================
# REQUEST
# ======================================================================

class AskRequest(BaseModel):

    question: str

    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
    )

    retrieve_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "question must be a non-empty string"
            )

        return value


# ======================================================================
# CITATION
# ======================================================================

class CitationResponse(BaseModel):

    source_id: str

    document_id: str

    chunk_index: int

    source_file: str | None = None

    case_name: str | None = None

    citation: str | None = None

    court: str | None = None

    judgment_date: str | None = None

    char_start: int

    char_end: int

    role: str


# ======================================================================
# RETRIEVED CHUNK
# ======================================================================

class RetrievedChunkResponse(BaseModel):

    rank: int

    score: float

    chunk_id: str

    document_id: str

    text: str

    primary_role: str

    roles: list[str] = Field(
        default_factory=list
    )

    chunk_index: int

    vector_score: float | None = None

    rerank_score: float | None = None

    case_name: str | None = None

    citation: str | None = None

    court: str | None = None

    judgment_date: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ======================================================================
# TIMING
# ======================================================================

class TimingResponse(BaseModel):

    retrieval: float = 0.0

    reranking: float = 0.0

    generation: float = 0.0

    total: float = 0.0


# ======================================================================
# ASK RESPONSE
# ======================================================================

class AskResponse(BaseModel):

    question: str

    answer: str

    citations: list[CitationResponse]

    retrieved_chunks: list[
        RetrievedChunkResponse
    ]

    model: str | None = None

    retrieval_count: int

    timing: TimingResponse

    debug: dict[str, Any] = Field(
        default_factory=dict
    )


# ======================================================================
# HEALTH
# ======================================================================

class HealthResponse(BaseModel):

    status: str

    qdrant: str

    ollama: str


# ======================================================================
# ERROR
# ======================================================================

class ErrorBody(BaseModel):

    code: str

    message: str


class ErrorResponse(BaseModel):

    error: ErrorBody