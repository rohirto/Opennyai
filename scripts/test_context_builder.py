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
# EXISTING RAG COMPONENTS
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


# =============================================================================
# CONFIGURATION
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


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG CONTEXT BUILDER TEST")
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
        f"Query type       : "
        f"{analysis.query_type}"
    )

    print(
        f"Preferred roles  : "
        f"{analysis.preferred_roles}"
    )

    print(
        f"Keywords         : "
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
    # 3. QDRANT STORE
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
        f"Metadata filter   : "
        f"{query_filter}"
    )

    # =========================================================================
    # 6. VECTOR RETRIEVAL
    # =========================================================================

    results = retriever.retrieve(
        query=QUERY,
        query_analysis=analysis,
        query_filter=query_filter,
        limit=RETRIEVAL_LIMIT,
    )

    print()
    print("=" * 80)
    print("VECTOR RETRIEVAL")
    print("=" * 80)

    print(
        f"Retrieved chunks  : "
        f"{len(results)}"
    )

    print()

    for rank, result in enumerate(
        results[:10],
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"score={result.vector_score:.6f} "
            f"chunk={result.chunk_index:3d} "
            f"role={result.primary_role:<15}"
        )

    # =========================================================================
    # 7. RERANKING
    # =========================================================================

    reranker = LegalReranker()

    reranked = reranker.rerank(
        results,
        analysis,
        top_k=RERANK_TOP_K,
    )

    print()
    print("=" * 80)
    print("ROLE-AWARE RERANKING")
    print("=" * 80)

    print(
        f"Reranked chunks   : "
        f"{len(reranked)}"
    )

    print()

    for rank, result in enumerate(
        reranked,
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"final={result.rerank_score:.6f} "
            f"vector={result.vector_score:.6f} "
            f"chunk={result.chunk_index:3d} "
            f"role={result.primary_role:<15}"
        )

    # =========================================================================
    # 8. CONTEXT BUILDER
    # =========================================================================

    context = assemble_context(
        results=reranked,
        query_analysis=analysis,
        max_context_chars=MAX_CONTEXT_CHARS,
    )

    # =========================================================================
    # 9. ASSEMBLED CONTEXT
    # =========================================================================

    print()
    print("=" * 80)
    print("ASSEMBLED CONTEXT")
    print("=" * 80)

    print()

    if context.text:
        print(
            context.text
        )
    else:
        print(
            "[NO CONTEXT WAS ASSEMBLED]"
        )

    # =========================================================================
    # 10. CONTEXT STATISTICS
    # =========================================================================

    print()
    print("=" * 80)
    print("CONTEXT STATISTICS")
    print("=" * 80)

    print(
        f"Characters       : "
        f"{context.character_count}"
    )

    print(
        f"Included chunks  : "
        f"{len(context.included_chunk_ids)}"
    )

    print(
        f"Skipped chunks   : "
        f"{len(context.skipped_chunk_ids)}"
    )

    print()

    print("Sections:")

    if context.sections:

        for section, count in (
            context.sections.items()
        ):

            print(
                f"  {section:<30}: "
                f"{count}"
            )

    else:

        print(
            "  [none]"
        )

    print()

    print("Included chunk IDs:")

    for chunk_id in (
        context.included_chunk_ids
    ):

        print(
            f"  {chunk_id}"
        )

    print()

    print("Skipped chunk IDs:")

    for chunk_id in (
        context.skipped_chunk_ids
    ):

        print(
            f"  {chunk_id}"
        )

    # =========================================================================
    # 11. LEGAL EVIDENCE CLASSIFICATION
    # =========================================================================

    print()
    print("=" * 80)
    print("LEGAL EVIDENCE CLASSIFICATION")
    print("=" * 80)

    classification = (
        context.debug.get(
            "classification",
            [],
        )
    )

    if classification:

        print()

        print(
            "Chunk  Role              "
            "Category                   "
            "Priority  Current  Other   Vector"
        )

        print(
            "-" * 80
        )

        for item in classification:

            print(
                f"{str(item['chunk_index']):<6} "
                f"{str(item['primary_role']):<18} "
                f"{str(item['category']):<25} "
                f"{item['priority']:<9.2f} "
                f"{item['current_court_outcome_signal']:<8.2f} "
                f"{item['other_court_outcome_signal']:<7.2f} "
                f"{item['vector_score']}"
            )

    else:

        print()
        print(
            "[NO CLASSIFICATION DEBUG DATA]"
        )

    # =========================================================================
    # 12. DECISION-QUERY SANITY CHECK
    # =========================================================================

    print()
    print("=" * 80)
    print("DECISION QUERY SANITY CHECK")
    print("=" * 80)

    categories = {
        item["category"]
        for item in classification
    }

    print()

    print(
        "Formal disposition present : "
        f"{'YES' if 'formal disposition' in categories else 'NO'}"
    )

    print(
        "Operative directions       : "
        f"{'YES' if 'operative directions' in categories else 'NO'}"
    )

    print(
        "Substantive holdings       : "
        f"{'YES' if 'substantive holding' in categories else 'NO'}"
    )

    print(
        "Lower-court outcome found  : "
        f"{'YES' if 'lower court outcome' in categories else 'NO'}"
    )

    print()

    # =========================================================================
    # 13. FINAL
    # =========================================================================

    print("=" * 80)
    print("CONTEXT BUILDER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()