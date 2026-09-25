"""
RAG ingestion stage.

RUN THIS SCRIPT WITH:
    .venv-rag

Input
-----
Artifacts produced by ingest_legacy.py:

output/
└── <document_name>/
    ├── cleaned_text.txt
    ├── ner.json
    ├── rhetorical_roles.json
    ├── summary_raw.json
    └── summary.txt

Pipeline
--------
OpenNyAI artifacts
        |
        v
LegalDocument
        |
        v
LegalChunk
        |
        v
BGE embeddings
        |
        v
Qdrant

Output
------
output/<document_name>/
    ├── document.json
    ├── chunks.json
    ├── chunk_debug.json
    ├── embeddings.npy
    └── rag_manifest.json

Qdrant
------
Collection:
    legal_documents

Vector:
    BAAI/bge-small-en-v1.5
    dimension = 384
    cosine similarity
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "output"


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

EMBEDDING_DEVICE = "auto"

EMBEDDING_BATCH_SIZE = 16

NORMALIZE_EMBEDDINGS = True

CHUNK_MAX_CHARS = 3000

CHUNK_MIN_STANDALONE_CHARS = 100

QDRANT_URL = (
    "http://localhost:6333"
)

QDRANT_COLLECTION = (
    "legal_documents"
)

VECTOR_SIZE = 384

QDRANT_BATCH_SIZE = 64

SOURCE_TYPE = "legal_judgment"

LANGUAGE = "en"


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from rag.ingestion.document_builder import (
    build_documents_from_file,
    document_statistics,
    save_document,
)

from rag.ingestion.chunker import (
    chunk_document,
    chunks_to_dict,
    export_chunk_debug,
    chunk_statistics,
)

from rag.ingestion.embedder import (
    LegalEmbedder,
    EmbeddingConfig,
)

from rag.ingestion.qdrant_ingest import (
    QdrantLegalStore,
)


# ============================================================================
# JSON HELPERS
# ============================================================================

def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    data: Any,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================================
# OPTIONAL METADATA
# ============================================================================

def load_metadata(
    document_name: str,
) -> dict[str, Any]:

    """
    Load optional trusted metadata.

    If present:

        metadata/<document_name>.json

    Example:

        {
            "case_name": "...",
            "citation": "...",
            "court": "...",
            "judgment_date": "..."
        }

    If absent, return an empty dictionary.

    The document builder deliberately does not attempt to infer this metadata
    from arbitrary text.
    """

    metadata_dir = (
        PROJECT_ROOT / "metadata"
    )

    metadata_path = (
        metadata_dir
        / f"{document_name}.json"
    )

    if not metadata_path.exists():

        return {}

    metadata = load_json(
        metadata_path
    )

    if not isinstance(
        metadata,
        dict,
    ):

        raise ValueError(
            f"Metadata file must contain "
            f"a JSON object: {metadata_path}"
        )

    return metadata


# ============================================================================
# PROCESS ONE DOCUMENT
# ============================================================================

def process_document(
    document_dir: Path,
    embedder: LegalEmbedder,
    qdrant_store: QdrantLegalStore,
) -> bool:

    start_time = time.perf_counter()

    document_name = (
        document_dir.name
    )

    print()
    print("=" * 90)
    print(
        f"RAG PROCESSING: {document_name}"
    )
    print("=" * 90)

    # ------------------------------------------------------------------------
    # Expected legacy artifacts
    # ------------------------------------------------------------------------

    rhetorical_roles_path = (
        document_dir
        / "rhetorical_roles.json"
    )

    ner_path = (
        document_dir
        / "ner.json"
    )

    summary_path = (
        document_dir
        / "summary_raw.json"
    )

    cleaned_text_path = (
        document_dir
        / "cleaned_text.txt"
    )

    required_files = [
        rhetorical_roles_path,
        ner_path,
        summary_path,
        cleaned_text_path,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing legacy artifacts:\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )

    # ------------------------------------------------------------------------
    # 1. Load OpenNyAI artifacts
    # ------------------------------------------------------------------------

    print()
    print(
        "[1/4] Loading OpenNyAI artifacts..."
    )

    ner_data = load_json(
        ner_path
    )

    summary_data = load_json(
        summary_path
    )

    metadata = load_metadata(
        document_name
    )

    print(
        f"RRC file     : "
        f"{rhetorical_roles_path.name}"
    )

    print(
        f"NER file     : "
        f"{ner_path.name}"
    )

    print(
        f"Summary file : "
        f"{summary_path.name}"
    )

    if metadata:

        print(
            "Metadata     : loaded"
        )

    else:

        print(
            "Metadata     : none"
        )

    # ------------------------------------------------------------------------
    # 2. Build canonical LegalDocument
    # ------------------------------------------------------------------------

    print()
    print(
        "[2/4] Building canonical LegalDocument..."
    )

    documents = build_documents_from_file(
        rhetorical_roles_path=rhetorical_roles_path,
        ner_data=ner_data,
        summary_data=summary_data,
        metadata=metadata,
        source_type=SOURCE_TYPE,
        language=LANGUAGE,
    )

    if not documents:

        raise RuntimeError(
            "No LegalDocument objects were "
            "produced from RRC output."
        )

    print(
        f"Documents produced : "
        f"{len(documents)}"
    )

    # Normally there is one PDF -> one RRC document.
    #
    # We process every returned document independently so that the code
    # remains correct if OpenNyAI output later contains multiple documents.
    for document_number, document in enumerate(
        documents,
        start=1,
    ):

        print()
        print(
            f"Canonical document "
            f"{document_number}/{len(documents)}"
        )

        print(
            f"Document ID : "
            f"{document.document_id}"
        )

        print(
            f"Text chars  : "
            f"{document.text_length:,}"
        )

        print(
            f"Annotations : "
            f"{document.annotation_count:,}"
        )

        print(
            f"Entities    : "
            f"{document.entity_count:,}"
        )

        # --------------------------------------------------------------------
        # Save canonical document
        # --------------------------------------------------------------------

        document_json_path = (
            document_dir
            / "document.json"
        )

        save_document(
            document,
            document_json_path,
        )

        # --------------------------------------------------------------------
        # Document diagnostics
        # --------------------------------------------------------------------

        statistics = document_statistics(
            document
        )

        save_json(
            statistics,
            document_dir
            / "document_statistics.json",
        )

        # --------------------------------------------------------------------
        # 3. Legal chunking
        # --------------------------------------------------------------------

        print()
        print(
            "[3/4] Creating legal chunks..."
        )

        chunks = chunk_document(
            document,
            max_chars=CHUNK_MAX_CHARS,
            min_standalone_chars=(
                CHUNK_MIN_STANDALONE_CHARS
            ),
        )

        if not chunks:

            raise RuntimeError(
                "Chunker returned zero chunks."
            )

        stats = chunk_statistics(chunks)

        print()
        print("Chunk statistics:")
        print(f"  Chunks created        : {stats['chunk_count']}")
        print(f"  Min chars             : {stats['min_chars']}")
        print(f"  Max chars             : {stats['max_chars']}")
        print(f"  Avg chars             : {stats['avg_chars']}")
        print(f"  Tiny chunks (<100)    : {stats['tiny_chunks_lt_100']}")
        print(f"  Structural fragments  : {stats.get('structural_fragments_lt_100', 0)}")
        print(f"  Chunks with entities  : {stats['chunks_with_entities']}")
        print(f"  Chunks with summary   : {stats['chunks_with_summary_context']}")
        print(f"  Roles                 : {stats['roles']}")

        # Save simple serialized chunks.
        chunks_json_path = (
            document_dir
            / "chunks.json"
        )

        save_json(
            chunks_to_dict(
                chunks
            ),
            chunks_json_path,
        )

        # Save detailed audit/debug information.
        export_chunk_debug(
            chunks,
            document_dir
            / "chunk_debug.json",
        )

        # --------------------------------------------------------------------
        # 4. Generate embeddings
        # --------------------------------------------------------------------

        print()
        print(
            "[4/4] Generating BGE embeddings..."
        )

        vectors = embedder.encode_chunks(
            chunks
        )

        print(
            f"Embedding shape : "
            f"{vectors.shape}"
        )

        if vectors.ndim != 2:

            raise RuntimeError(
                f"Expected 2D embeddings, "
                f"got shape {vectors.shape}"
            )

        if vectors.shape[0] != len(
            chunks
        ):

            raise RuntimeError(
                "Embedding count does not "
                "match chunk count."
            )

        if vectors.shape[1] != VECTOR_SIZE:

            raise RuntimeError(
                f"Expected embedding dimension "
                f"{VECTOR_SIZE}, "
                f"got {vectors.shape[1]}"
            )

        embeddings_path = (
            document_dir
            / "embeddings.npy"
        )

        # numpy is already returned by the embedder.
        import numpy as np

        np.save(
            embeddings_path,
            vectors,
        )

        # --------------------------------------------------------------------
        # Qdrant ingestion
        # --------------------------------------------------------------------

        print()
        print(
            "Uploading chunks to Qdrant..."
        )

        qdrant_result = qdrant_store.ingest(
            chunks=chunks,
            vectors=vectors,
            document=document,
            replace_document=True,
            batch_size=QDRANT_BATCH_SIZE,
        )

        print(
            f"Qdrant result : "
            f"{qdrant_result}"
        )

        # --------------------------------------------------------------------
        # Manifest
        # --------------------------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        manifest = {
            "document_name": document_name,
            "document_id": document.document_id,
            "source_file": document.source_file,
            "source_type": document.source_type,
            "language": document.language,
            "case_name": document.case_name,
            "citation": document.citation,
            "court": document.court,
            "judgment_date": document.judgment_date,
            "text_length": document.text_length,
            "annotation_count": document.annotation_count,
            "entity_count": document.entity_count,
            "chunk_count": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": VECTOR_SIZE,
            "embedding_normalized": NORMALIZE_EMBEDDINGS,
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "chunk_min_standalone_chars": (
                CHUNK_MIN_STANDALONE_CHARS
            ),
            "qdrant_url": QDRANT_URL,
            "qdrant_collection": QDRANT_COLLECTION,
            "elapsed_seconds": round(
                elapsed,
                3,
            ),
            "status": "success",
        }

        save_json(
            manifest,
            document_dir
            / "rag_manifest.json",
        )

    # ------------------------------------------------------------------------
    # Final document summary
    # ------------------------------------------------------------------------

    print()
    print("-" * 90)
    print("RAG INGESTION COMPLETE")
    print("-" * 90)

    print(
        f"Document : {document_name}"
    )

    print(
        f"Elapsed  : {elapsed:.2f} sec"
    )

    print()
    print("Artifacts:")

    print(
        f"  {document_dir / 'document.json'}"
    )

    print(
        f"  {document_dir / 'document_statistics.json'}"
    )

    print(
        f"  {document_dir / 'chunks.json'}"
    )

    print(
        f"  {document_dir / 'chunk_debug.json'}"
    )

    print(
        f"  {document_dir / 'embeddings.npy'}"
    )

    print(
        f"  {document_dir / 'rag_manifest.json'}"
    )

    return True


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print()
    print("=" * 90)
    print("LEGAL RAG INGESTION")
    print("=" * 90)

    print()
    print(
        f"Project root       : {PROJECT_ROOT}"
    )

    print(
        f"Output directory   : {OUTPUT_DIR}"
    )

    print(
        f"Embedding model    : {EMBEDDING_MODEL}"
    )

    print(
        f"Embedding device   : {EMBEDDING_DEVICE}"
    )

    print(
        f"Qdrant             : {QDRANT_URL}"
    )

    print(
        f"Collection         : {QDRANT_COLLECTION}"
    )

    print(
        f"Vector dimension   : {VECTOR_SIZE}"
    )

    print(
        f"Chunk max chars    : {CHUNK_MAX_CHARS}"
    )

    if not OUTPUT_DIR.exists():

        print()
        print(
            f"ERROR: Output directory does not exist: "
            f"{OUTPUT_DIR}"
        )

        return 1

    # ------------------------------------------------------------------------
    # Initialize BGE
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("INITIALIZING EMBEDDER")
    print("=" * 90)

    embedder = LegalEmbedder(
        EmbeddingConfig(
            model_name=EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=(
                NORMALIZE_EMBEDDINGS
            ),
            show_progress_bar=False,
        )
    )

    # LegalEmbedder stores the underlying model dimension
    # in self.dimension.
    actual_dimension = embedder.dimension

    print(
        f"Embedding dimension: "
        f"{actual_dimension}"
    )

    if actual_dimension != VECTOR_SIZE:

        raise RuntimeError(
            f"Configured vector size "
            f"{VECTOR_SIZE} does not match "
            f"embedder dimension "
            f"{actual_dimension}."
        )

    # ------------------------------------------------------------------------
    # Initialize Qdrant
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("INITIALIZING QDRANT")
    print("=" * 90)

    qdrant_store = QdrantLegalStore(
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        vector_size=embedder.dimension,
        embedding_model=EMBEDDING_MODEL,
    )

    if not qdrant_store.health_check():

        print()
        print(
            "ERROR: Qdrant health check failed."
        )

        print(
            f"Expected Qdrant at: "
            f"{QDRANT_URL}"
        )

        return 1

    print(
        "Qdrant health check: OK"
    )

    # Do NOT recreate the collection here.
    #
    # This allows the script to be rerun without destroying previously
    # indexed documents.
    qdrant_store.create_collection(
        recreate=True
    )

    # ------------------------------------------------------------------------
    # Find legacy-processed documents
    # ------------------------------------------------------------------------

    REQUIRED_LEGACY_ARTIFACTS = {
        "rhetorical_roles.json",
        "ner.json",
        "summary_raw.json",
        "cleaned_text.txt",
    }

    document_dirs = []

    for path in OUTPUT_DIR.iterdir():

        if not path.is_dir():
            continue

        existing = {
            p.name
            for p in path.iterdir()
            if p.is_file()
        }

        if REQUIRED_LEGACY_ARTIFACTS.issubset(existing):
            document_dirs.append(path)
        else:
            print(
                f"Skipping non-document directory: {path.name}"
            )

    document_dirs.sort()

    if not document_dirs:

        print()
        print(
            "No document directories found."
        )

        print(
            "Run ingest_legacy.py first."
        )

        return 0

    print()
    print(
        f"Found {len(document_dirs)} "
        f"document directory/directories."
    )

    successful = 0
    failed = 0

    # ------------------------------------------------------------------------
    # Process documents
    # ------------------------------------------------------------------------

    for index, document_dir in enumerate(
        document_dirs,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(document_dirs)}]"
        )

        try:

            process_document(
                document_dir=document_dir,
                embedder=embedder,
                qdrant_store=qdrant_store,
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print()
            print("!" * 90)

            print(
                f"FAILED: "
                f"{document_dir.name}"
            )

            print("!" * 90)

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("RAG INGESTION SUMMARY")
    print("=" * 90)

    print(
        f"Total documents : "
        f"{len(document_dirs)}"
    )

    print(
        f"Successful      : "
        f"{successful}"
    )

    print(
        f"Failed          : "
        f"{failed}"
    )

    print()

    if failed:

        print(
            "STATUS: COMPLETED WITH ERRORS"
        )

        return 1

    print(
        "STATUS: SUCCESS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )