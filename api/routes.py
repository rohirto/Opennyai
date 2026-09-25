# api/routes.py

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from .models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    HealthResponse,
    RetrievedChunkResponse,
    TimingResponse,
)


router = APIRouter()


# ======================================================================
# HELPERS
# ======================================================================

def get_pipeline(request: Request):

    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None,
    )

    if pipeline is None:
        raise RuntimeError(
            "RAG pipeline is not initialized"
        )

    return pipeline


def result_to_chunk(
    result,
    rank: int,
) -> RetrievedChunkResponse:

    score = (
        result.rerank_score
        if result.rerank_score is not None
        else result.vector_score
        if result.vector_score is not None
        else result.score
    )

    return RetrievedChunkResponse(

        rank=rank,

        score=float(score),

        chunk_id=result.chunk_id,

        document_id=result.document_id,

        text=result.text,

        primary_role=result.primary_role,

        roles=result.roles,

        chunk_index=result.chunk_index,

        vector_score=result.vector_score,

        rerank_score=result.rerank_score,

        case_name=result.case_name,

        citation=result.citation,

        court=result.court,

        judgment_date=result.judgment_date,

        metadata=result.metadata or {},
    )


# ======================================================================
# HEALTH
# ======================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    request: Request,
) -> HealthResponse:

    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None,
    )

    if pipeline is None:

        return HealthResponse(
            status="degraded",
            qdrant="unknown",
            ollama="unknown",
        )

    # --------------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------------

    qdrant_status = "ok"

    try:

        pipeline.retriever.qdrant_store.client.get_collections()

    except Exception:

        qdrant_status = "unavailable"

    # --------------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------------

    ollama_status = "ok"

    try:

        health_check = getattr(
            pipeline.llm,
            "health_check",
            None,
        )

        if callable(health_check):

            if not health_check():
                ollama_status = "unavailable"

    except Exception:

        ollama_status = "unavailable"

    # --------------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------------

    status = (
        "ok"
        if (
            qdrant_status == "ok"
            and
            ollama_status == "ok"
        )
        else "degraded"
    )

    return HealthResponse(

        status=status,

        qdrant=qdrant_status,

        ollama=ollama_status,
    )


# ======================================================================
# ASK
# ======================================================================

@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
    http_request: Request,
) -> AskResponse:

    pipeline = get_pipeline(
        http_request
    )

    started = time.perf_counter()

    # --------------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------------

    result = pipeline.ask(

        query=request.question,

        retrieve_k=request.retrieve_k,

        final_k=request.top_k,
    )

    total_time = (
        time.perf_counter()
        - started
    )

    # --------------------------------------------------------------
    # CITATIONS
    # --------------------------------------------------------------

    citations = []

    for source in result.sources:

        citations.append(
            CitationResponse(

                source_id=source.source_id,

                document_id=source.document_id,

                chunk_index=source.chunk_index,

                source_file=source.source_file,

                case_name=source.case_name,

                citation=source.citation,

                court=source.court,

                judgment_date=source.judgment_date,

                char_start=source.char_start,

                char_end=source.char_end,

                role=source.role,
            )
        )

    # --------------------------------------------------------------
    # RETRIEVED CHUNKS
    # --------------------------------------------------------------

    retrieved_chunks = []

    for rank, chunk in enumerate(
        result.retrieved_results,
        start=1,
    ):

        retrieved_chunks.append(
            result_to_chunk(
                chunk,
                rank,
            )
        )

    # --------------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------------

    return AskResponse(

        question=result.query,

        answer=result.answer,

        citations=citations,

        retrieved_chunks=retrieved_chunks,

        model=result.model,

        retrieval_count=result.retrieval_count,

        timing=TimingResponse(

            retrieval=0.0,

            reranking=0.0,

            generation=total_time,

            total=total_time,
        ),

        debug=result.debug,
    )