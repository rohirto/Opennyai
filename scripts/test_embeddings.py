from pathlib import Path
import json
import sys

import numpy as np

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ingestion.embedder import (
    LegalEmbedder,
    EmbeddingConfig,
)


CHUNKS_FILE = (
    ROOT
    / "output"
    / "Sunil_Kumar_B"
    / "chunk_debug.json"
)


def load_chunks(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)




def main():

    print("=" * 80)
    print("EMBEDDING TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load chunks
    # ---------------------------------------------------------

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{CHUNKS_FILE}"
        )

    chunks = load_chunks(CHUNKS_FILE)

    print(f"Chunks loaded: {len(chunks)}")

    # ---------------------------------------------------------
    # Initialize model
    # ---------------------------------------------------------

    embedder = LegalEmbedder(
        EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            device="auto",
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )

    # ---------------------------------------------------------
    # Build texts
    # ---------------------------------------------------------

    texts = []

    for chunk in chunks:

        embedding_text = (
            chunk.get("embedding_text")
            or chunk.get("text")
            or ""
        )

        texts.append(
            embedding_text.strip()
        )

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    embeddings = embedder.encode(texts)

    print()
    print("Embedding result:")
    print(f"Shape      : {embeddings.shape}")
    print(f"Dtype      : {embeddings.dtype}")
    print(f"Dimension  : {embeddings.shape[1]}")

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    assert embeddings.shape[0] == len(chunks)

    assert embeddings.shape[1] == (
        embedder.dimension
    )

    assert embeddings.dtype == np.float32

    # Check NaN / Inf
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()

    # Check normalization
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print()
    print("Norm statistics:")
    print(f"Min: {norms.min():.6f}")
    print(f"Max: {norms.max():.6f}")
    print(f"Avg: {norms.mean():.6f}")

    # ---------------------------------------------------------
    # Similarity sanity check
    # ---------------------------------------------------------

    query = (
        "What did the Supreme Court decide "
        "about the 2014 amendment to the "
        "Employees Pension Scheme?"
    )

    query_embedding = (
        embedder.encode_query(query)
    )

    similarities = (
        embeddings @ query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:5]

    print()
    print("=" * 80)
    print("TOP 5 SEMANTIC MATCHES")
    print("=" * 80)

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        chunk = chunks[index]

        print()
        print(
            f"Rank {rank}"
        )

        print(
            f"Chunk : "
            f"{chunk.get('chunk_index', index)}"
        )

        print(
            f"Score : "
            f"{similarities[index]:.4f}"
        )

        print(
            f"Role  : "
            f"{chunk.get('primary_role')}"
        )

        text = (
            chunk.get("text", "")
            .replace("\n", " ")
        )

        print(
            f"Text  : "
            f"{text[:500]}"
        )

    print()
    print("=" * 80)
    print("EMBEDDING TEST PASSED")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # Save embeddings
    # ---------------------------------------------------------

    JUDGMENT_NAME = CHUNKS_FILE.parent.name

    EMBEDDINGS_DIR = (
        ROOT
        / "output"
        / "embeddings"
        / JUDGMENT_NAME
    )

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EMBEDDINGS_FILE = (
        EMBEDDINGS_DIR
        / "embeddings.npy"
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    print(
        f"\nSaved embeddings to:\n"
        f"{EMBEDDINGS_FILE}"
    )

    import json
    
    embedding_metadata = {
        "judgment": JUDGMENT_NAME,
        "chunks_file": str(CHUNKS_FILE),
        "embedding_file": str(EMBEDDINGS_FILE),
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "normalized": True,
        "chunks": [
            {
                "row": i,
                "chunk_index": chunk.get(
                    "chunk_index",
                    i,
                ),
                "document_id": chunk.get(
                    "document_id"
                ),
                "char_start": chunk.get(
                    "char_start"
                ),
                "char_end": chunk.get(
                    "char_end"
                ),
            }
            for i, chunk in enumerate(chunks)
        ],
    }

    METADATA_FILE = (
        EMBEDDINGS_DIR
        / "embedding_metadata.json"
    )

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            embedding_metadata,
            f,
            indent=2,
        )

    print(
        f"Saved embedding metadata to:\n"
        f"{METADATA_FILE}"
    )



if __name__ == "__main__":
    main()