from __future__ import annotations

from pathlib import Path
import sys


# =============================================================================
# PROJECT ROOT
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# IMPORTS
# =============================================================================

from rag.ingestion.embedder import (
    LegalEmbedder,
    EmbeddingConfig,
)

from rag.ingestion.qdrant_ingest import (
    QdrantLegalStore,
)

from rag.retrieval.query_classifier import (
    LegalQueryClassifier,
)

from rag.retrieval.metadata_filter import (
    LegalMetadataFilterBuilder,
)

from rag.retrieval.retriever import (
    LegalRetriever,
)

from rag.retrieval.reranker import (
    LegalReranker,
)

from rag.generation.context_builder import (
    assemble_context,
)

from rag.generation.llm import (
    OllamaConfig,
    OllamaLegalLLM,
)

from rag.generation.generator import (
    LegalAnswerGenerator,
)


# =============================================================================
# CONFIG
# =============================================================================

QUERY = (
    "What did the Supreme Court decide "
    "about the 2014 amendment to the "
    "Employees Pension Scheme?"
)

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "legal_documents"

VECTOR_SIZE = 384

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RETRIEVAL_LIMIT = 20

RERANK_TOP_K = 10

MAX_CONTEXT_CHARS = 12000

OLLAMA_URL = "http://localhost:11434"

OLLAMA_MODEL = "qwen2.5:3b-instruct"


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG GENERATION TEST")
    print("=" * 80)

    # =========================================================================
    # 1. QUERY CLASSIFICATION
    # =========================================================================

    classifier = LegalQueryClassifier()

    analysis = classifier.classify(
        QUERY
    )

    print()
    print("QUERY ANALYSIS")
    print("-" * 80)

    print(
        f"Query type      : "
        f"{analysis.query_type}"
    )

    print(
        f"Preferred roles : "
        f"{analysis.preferred_roles}"
    )

    print(
        f"Keywords        : "
        f"{analysis.keywords}"
    )

    # =========================================================================
    # 2. EMBEDDER
    # =========================================================================

    embedder = LegalEmbedder(
        EmbeddingConfig(
            model_name=EMBEDDING_MODEL,
            device="auto",
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    # =========================================================================
    # 3. QDRANT
    # =========================================================================

    store = QdrantLegalStore(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_SIZE,
        embedding_model=EMBEDDING_MODEL,
    )

    # =========================================================================
    # 4. RETRIEVER
    # =========================================================================

    retriever = LegalRetriever(
        qdrant_store=store,
        embedder=embedder,
    )

    # =========================================================================
    # 5. METADATA FILTER
    # =========================================================================

    filter_builder = (
        LegalMetadataFilterBuilder()
    )

    query_filter = (
        filter_builder.build(
            analysis
        )
    )

    print()
    print(
        f"Metadata filter  : "
        f"{query_filter}"
    )

    # =========================================================================
    # 6. RETRIEVE
    # =========================================================================

    results = retriever.retrieve(
        query=QUERY,
        query_analysis=analysis,
        query_filter=query_filter,
        limit=RETRIEVAL_LIMIT,
    )

    print()
    print(
        f"Retrieved        : "
        f"{len(results)}"
    )

    # =========================================================================
    # 7. RERANK
    # =========================================================================

    reranker = LegalReranker()

    reranked = reranker.rerank(
        results,
        analysis,
        top_k=RERANK_TOP_K,
    )

    print(
        f"Reranked         : "
        f"{len(reranked)}"
    )

    # =========================================================================
    # 8. BUILD CONTEXT
    # =========================================================================

    context = assemble_context(
        results=reranked,
        query_analysis=analysis,
        max_context_chars=MAX_CONTEXT_CHARS,
    )

    print()
    print("=" * 80)
    print("CONTEXT")
    print("=" * 80)

    print()

    print(
        context.text
    )

    # =========================================================================
    # 9. OLLAMA
    # =========================================================================

    llm = OllamaLegalLLM(
        OllamaConfig(
            base_url=OLLAMA_URL,
            model_name=OLLAMA_MODEL,
            temperature=0.1,
            top_p=0.9,
            num_predict=1200,
            timeout=180,
        )
    )

    print()
    print("=" * 80)
    print("OLLAMA")
    print("=" * 80)

    if not llm.health_check():

        print()
        print(
            "ERROR: Ollama is not reachable at "
            f"{OLLAMA_URL}"
        )

        print()
        print(
            "Start Ollama and make sure the model exists:"
        )

        print(
            f"    ollama list"
        )

        print(
            f"    ollama run {OLLAMA_MODEL}"
        )

        return

    print(
        f"Ollama URL       : "
        f"{OLLAMA_URL}"
    )

    print(
        f"Model            : "
        f"{OLLAMA_MODEL}"
    )

    # =========================================================================
    # 10. GENERATE
    # =========================================================================

    generator = LegalAnswerGenerator(
        llm=llm,
    )

    generated = generator.generate(
        query=QUERY,
        query_analysis=analysis,
        context=context,
        results=reranked,
    )

    # =========================================================================
    # 11. ANSWER
    # =========================================================================

    print()
    print("=" * 80)
    print("GENERATED LEGAL ANSWER")
    print("=" * 80)

    print()

    print(
        generated.answer
    )

    # =========================================================================
    # 12. SOURCES
    # =========================================================================

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    print()

    for source in generated.sources:

        print(
            f"Chunk {source.chunk_index} | "
            f"Role={source.role} | "
            f"Citation={source.citation}"
        )

    # =========================================================================
    # 13. DEBUG
    # =========================================================================

    print()
    print("=" * 80)
    print("GENERATION DEBUG")
    print("=" * 80)

    for key, value in (
        generated.debug.items()
    ):

        print(
            f"{key:<25}: {value}"
        )

    print()
    print("=" * 80)
    print("GENERATION TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()