from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_documents"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 10


# ---------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------

QUERY = (
    "What did the Supreme Court decide "
    "about the 2014 amendment to the "
    "Employees Pension Scheme?"
)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("QDRANT RETRIEVAL TEST")
    print("=" * 80)

    print(f"\nQuery:\n{QUERY}\n")

    # ---------------------------------------------------------------
    # Connect to Qdrant
    # ---------------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    print(
        f"[Qdrant] Connected: {QDRANT_URL}"
    )

    # ---------------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------------

    print(
        f"[Embedding] Loading: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
        if __import__("torch").cuda.is_available()
        else "cpu",
    )

    print(
        f"[Embedding] Device: "
        f"{model.device}"
    )

    # ---------------------------------------------------------------
    # Encode query
    # ---------------------------------------------------------------

    query_vector = model.encode(
        QUERY,
        normalize_embeddings=True,
    )

    print(
        f"[Embedding] Query dimension: "
        f"{len(query_vector)}"
    )

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True,
    ).points

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print(f"TOP {TOP_K} RETRIEVED CHUNKS")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = result.payload or {}

        print()
        print("-" * 80)

        print(
            f"Rank        : {rank}"
        )

        print(
            f"Score       : {result.score:.6f}"
        )

        print(
            f"Chunk       : "
            f"{payload.get('chunk_index')}"
        )

        print(
            f"Primary Role: "
            f"{payload.get('primary_role')}"
        )

        print(
            f"Roles       : "
            f"{payload.get('roles')}"
        )

        print(
            f"Characters  : "
            f"{payload.get('char_start')} - "
            f"{payload.get('char_end')}"
        )

        print(
            f"Entities    : "
            f"{payload.get('entities')}"
        )

        print(
            "\nTEXT:\n"
        )

        print(
            payload.get(
                "text",
                "",
            )
        )

    print()
    print("=" * 80)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()