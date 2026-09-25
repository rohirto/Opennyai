# api/main.py

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles

from rag.generation.llm import (
    OllamaConfig,
    OllamaLegalLLM,
)

from rag.ingestion.embedder import (
    EmbeddingConfig,
    LegalEmbedder,
)

from rag.ingestion.qdrant_ingest import (
    QdrantLegalStore,
)

from rag.retrieval.retriever import (
    LegalRetriever,
)

from rag.retrieval.reranker import (
    LegalReranker,
)

from rag.pipeline import (
    LegalRAGPipeline,
)

from .models import (
    ErrorResponse,
)

from .routes import (
    router,
)


# ======================================================================
# LOGGING
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(
    "epfo_legal_api"
)


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(
    __file__
).resolve().parent

STATIC_DIR = (
    BASE_DIR
    / "static"
)


# ======================================================================
# PIPELINE FACTORY
# ======================================================================

def create_rag_pipeline():

    logger.info(
        "Initializing Legal RAG pipeline..."
    )

    # --------------------------------------------------------------
    # EMBEDDER
    # --------------------------------------------------------------

    embedder = LegalEmbedder(

        EmbeddingConfig(

            model_name=(
                "BAAI/"
                "bge-small-en-v1.5"
            ),

            device="auto",

            batch_size=16,

            normalize_embeddings=True,

            show_progress_bar=False,
        )
    )

    # --------------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------------

    store = QdrantLegalStore(

        url=(
            "http://localhost:6333"
        ),

        collection_name=(
            "legal_documents"
        ),

        vector_size=384,

        embedding_model=(
            "BAAI/"
            "bge-small-en-v1.5"
        ),
    )

    # --------------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------------

    retriever = LegalRetriever(

        qdrant_store=store,

        embedder=embedder,
    )

    # --------------------------------------------------------------
    # RERANKER
    # --------------------------------------------------------------

    reranker = LegalReranker()

    # --------------------------------------------------------------
    # LLM
    # --------------------------------------------------------------
    from rag.generation.llm_factory import create_llm
    llm = create_llm()
    

    # --------------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------------

    pipeline = LegalRAGPipeline(

        embedder=embedder,

        retriever=retriever,

        reranker=reranker,

        llm=llm,
    )

    logger.info(
        "Legal RAG pipeline initialized"
    )

    return pipeline


# ======================================================================
# LIFESPAN
# ======================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "Starting EPFO Legal Assistant..."
    )

    app.state.rag_pipeline = (
        create_rag_pipeline()
    )

    yield

    logger.info(
        "Shutting down EPFO Legal Assistant..."
    )


# ======================================================================
# APP
# ======================================================================

app = FastAPI(

    title=(
        "EPFO Legal RAG API"
    ),

    version="1.0.0",

    lifespan=lifespan,
)


app.include_router(
    router
)


# ======================================================================
# STATIC FILES
# ======================================================================

# Keep the supplied frontend untouched.
app.mount(

    "/static",

    StaticFiles(
        directory=STATIC_DIR
    ),

    name="static",
)


@app.get(
    "/",
    include_in_schema=False,
)
def serve_frontend():

    return FileResponse(
        STATIC_DIR
        / "index.html"
    )


# ======================================================================
# ERRORS
# ======================================================================

def error_response(
    status_code: int,
    code: str,
    message: str,
):

    body = (
        ErrorResponse(
            error={
                "code": code,
                "message": message,
            }
        )
        .model_dump()
    )

    return JSONResponse(

        status_code=status_code,

        content=body,
    )


@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    return error_response(

        422,

        "validation_error",

        "The request is invalid.",
    )


@app.exception_handler(
    RuntimeError
)
async def runtime_exception_handler(
    request: Request,
    exc: RuntimeError,
):

    message = str(exc)

    if (
        "Qdrant"
        in message
        and
        "reach"
        in message.lower()
    ):

        return error_response(

            503,

            "qdrant_unavailable",

            "The retrieval service is unavailable.",
        )

    logger.exception(
        "Runtime error"
    )

    return error_response(

        500,

        "internal_error",

        "The request could not be completed.",
    )


@app.exception_handler(
    Exception
)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled API error"
    )

    return error_response(

        500,

        "internal_error",

        "The request could not be completed.",
    )