from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rag.ingestion.qdrant_ingest import (
    QdrantLegalStore,
)


JUDGMENT = "Sunil_Kumar_B"
DOCUMENT_ID = (
    "ae13fdffdb88559c50822e8a6df1ee04be868a28086fa4ac812fd6205e718e7f"
)


CHUNKS_FILE = (
    ROOT
    / "output"
    / JUDGMENT
    / "chunk_debug.json"
)


EMBEDDINGS_FILE = (
    ROOT
    / "output"
    / "embeddings"
    / JUDGMENT
    / "embeddings.npy"
)


def main():

    print("=" * 80)
    print("QDRANT LEGAL INGESTION")
    print("=" * 80)

    print()
    print(f"Judgment   : {JUDGMENT}")
    print(f"Chunks     : {CHUNKS_FILE}")
    print(f"Embeddings : {EMBEDDINGS_FILE}")
    print()

    # --------------------------------------------------------------
    # Store
    # --------------------------------------------------------------

    store = QdrantLegalStore(
        url="http://localhost:6333",
        collection_name="legal_documents",
        vector_size=384,
        embedding_model=(
            "BAAI/bge-small-en-v1.5"
        ),
    )

    # --------------------------------------------------------------
    # Health
    # --------------------------------------------------------------

    if not store.health_check():

        raise RuntimeError(
            "Qdrant is not reachable."
        )

    # --------------------------------------------------------------
    # Collection
    # --------------------------------------------------------------

    store.create_collection(
        recreate=False
    )

    # --------------------------------------------------------------
    # Ingest
    # --------------------------------------------------------------

    store.ingest_from_files(
        chunks_file=CHUNKS_FILE,
        embeddings_file=EMBEDDINGS_FILE,
        document_id=DOCUMENT_ID,
        batch_size=64,
    )

    # --------------------------------------------------------------
    # Information
    # --------------------------------------------------------------

    info = store.info()

    print()
    print("=" * 80)
    print("QDRANT COLLECTION")
    print("=" * 80)

    print(info)

    print()
    print("=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()