from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


QUERY = (
    "What did the Supreme Court decide "
    "about the 2014 amendment to the "
    "Employees Pension Scheme?"
)


def main():

    print("=" * 80)
    print("LEGAL RAG RETRIEVAL PIPELINE TEST")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Query classification
    # -----------------------------------------------------------------

    classifier = LegalQueryClassifier()

    analysis = classifier.classify(
        QUERY
    )

    print("\nQUERY ANALYSIS")
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

    # -----------------------------------------------------------------
    # Components
    # -----------------------------------------------------------------

    embedder = LegalEmbedder(
        EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            device="auto",
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    store = QdrantLegalStore(
        url="http://localhost:6333",
        collection_name="legal_documents",
        vector_size=384,
        embedding_model="BAAI/bge-small-en-v1.5",
    )

    retriever = LegalRetriever(
        qdrant_store=store,
        embedder=embedder,
    )

    filter_builder = (
        LegalMetadataFilterBuilder()
    )

    reranker = LegalReranker()

    # -----------------------------------------------------------------
    # Metadata filter
    # -----------------------------------------------------------------

    query_filter = (
        filter_builder.build(
            analysis
        )
    )

    print(
        f"\nMetadata filter   : "
        f"{query_filter}"
    )

    # -----------------------------------------------------------------
    # Vector retrieval
    # -----------------------------------------------------------------

    results = retriever.retrieve(
        query=QUERY,
        query_analysis=analysis,
        query_filter=query_filter,
        limit=20,
    )

    print()
    print("=" * 80)
    print("VECTOR RETRIEVAL")
    print("=" * 80)

    for rank, result in enumerate(
        results[:10],
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"score={result.vector_score:.6f} "
            f"chunk={result.chunk_index:3d} "
            f"role={result.primary_role:<15} "
            f"case={result.case_name}"
        )

    # -----------------------------------------------------------------
    # Reranking
    # -----------------------------------------------------------------

    reranked = reranker.rerank(
        results,
        analysis,
        top_k=10,
    )

    print()
    print("=" * 80)
    print("ROLE-AWARE RERANKING")
    print("=" * 80)

    for rank, result in enumerate(
        reranked,
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"final={result.rerank_score:.6f} "
            f"vector={result.vector_score:.6f} "
            f"chunk={result.chunk_index:3d} "
            f"role={result.primary_role:<15} "
            f"case={result.case_name}"
        )

    # -----------------------------------------------------------------
    # Show retrieved text
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("RERANKED LEGAL CONTEXT")
    print("=" * 80)

    for rank, result in enumerate(
        reranked,
        start=1,
    ):

        print()
        print("-" * 80)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Chunk: {result.chunk_index}"
        )

        print(
            f"Role: {result.primary_role}"
        )

        print(
            f"Vector score: "
            f"{result.vector_score:.6f}"
        )

        print(
            f"Rerank score: "
            f"{result.rerank_score:.6f}"
        )

        print()

        print(
            result.text
        )

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()