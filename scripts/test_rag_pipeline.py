
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.generation.llm import (
    OllamaConfig,
    OllamaLegalLLM,
)

from rag.generation.generator import (
    LegalAnswerGenerator,
)

from rag.ingestion.embedder import (
    LegalEmbedder,
    EmbeddingConfig,
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

from rag.retrieval.query_classifier import (
    LegalQueryClassifier,
)

from rag.pipeline import (
    LegalRAGPipeline,
)


QUERY = (
    "What did the Supreme Court decide "
    "about the 2014 amendment to the "
    "Employees Pension Scheme?"
)


def main():

    print("=" * 80)
    print("END-TO-END LEGAL RAG TEST")
    print("=" * 80)

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

    reranker = LegalReranker()

    llm = OllamaLegalLLM(
        OllamaConfig(
            base_url="http://localhost:11434",
            model_name="qwen2.5:3b-instruct",
            temperature=0.1,
            top_p=0.9,
            num_predict=1200,
            timeout=180,
            keep_alive="10m",
        )
    )

    pipeline = LegalRAGPipeline(
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )

    print("\nQUERY")
    print("-" * 80)
    print(QUERY)

    print("\nRunning RAG pipeline...\n")

    result = pipeline.ask(
        query=QUERY,
        retrieve_k=20,
        final_k=8,
        max_context_chars=12000,
    )
    
    print()
    print("=" * 80)
    print("LLM CONTEXT")
    print("=" * 80)
    print()

    print(
        result.debug.get(
            "llm_context",
            "[No context available]"
        )
    )

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print()

    print(result.answer)

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result.sources:

        print(
            f"{source.source_id} | "
            f"chunk={source.chunk_index} | "
            f"role={source.role} | "
            f"case={source.case_name}"
        )

    print()
    print("=" * 80)
    print("DEBUG")
    print("=" * 80)

    for key, value in result.debug.items():

        print(f"{key}: {value}")


if __name__ == "__main__":
    main()