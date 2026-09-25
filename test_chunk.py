# scripts/test_rhetorical_role.py
from pathlib import Path
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from rag.ingestion.document_builder import (
    build_documents_from_file,
    document_statistics,
)

from rag.ingestion.chunker import (
    chunk_document,
    chunk_statistics,
    validate_chunks,
    export_chunk_debug,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = PROJECT_ROOT

RRC_PATH = BASE_DIR / "Opennyai"/ "output" / "Sunil_Kumar_B" / "rhetorical_roles.json"
NER_PATH = BASE_DIR/"Opennyai" /"output"/ "Sunil_Kumar_B" / "ner.json"
SUMMARY_PATH = BASE_DIR /"Opennyai"/ "output" / "Sunil_Kumar_B" / "summarizer.json"

OUTPUT_DIR = BASE_DIR /"Opennyai"/ "output" /"Sunil_Kumar_B" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load OpenNyAI outputs
# ---------------------------------------------------------------------

import json

with NER_PATH.open("r", encoding="utf-8") as f:
    ner_data = json.load(f)

with SUMMARY_PATH.open("r", encoding="utf-8") as f:
    summary_data = json.load(f)


# ---------------------------------------------------------------------
# Build LegalDocument
# ---------------------------------------------------------------------

documents = build_documents_from_file(
    rhetorical_roles_path=RRC_PATH,
    ner_data=ner_data,
    summary_data=summary_data,
    metadata={
        "case_name": "Employees Provident Fund Organisation & Anr. v. Sunil Kumar B. & Ors.",
        "citation": "2022 INSC 1171",
        "court": "Supreme Court of India",
        "judgment_date": "04 November 2022",
    },
)


print("=" * 80)
print("DOCUMENT BUILDER")
print("=" * 80)

print(f"Documents: {len(documents)}")

for document in documents:

    print("\nDocument:")
    print(f"  ID              : {document.document_id}")
    print(f"  Text length     : {document.text_length}")
    print(f"  RRC annotations : {document.annotation_count}")
    print(f"  NER entities    : {document.entity_count}")

    print("\nDocument statistics:")
    print(document_statistics(document))


# ---------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------

for document in documents:

    print("\n")
    print("=" * 80)
    print("CHUNKING")
    print("=" * 80)

    chunks = chunk_document(
        document,
        max_chars=2200,
        min_standalone_chars=100,
    )

    # -----------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------

    validate_chunks(
        document,
        chunks,
    )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    stats = chunk_statistics(chunks)

    print("\nChunk statistics:")
    print(stats)

    # -----------------------------------------------------------------
    # Print chunks
    # -----------------------------------------------------------------

    for chunk in chunks:

        print("\n" + "-" * 80)

        print(
            f"Chunk {chunk.chunk_index}"
        )

        print(
            f"Role       : {chunk.primary_role}"
        )

        print(
            f"Roles      : {chunk.roles}"
        )

        print(
            f"Characters : {chunk.char_start} - {chunk.char_end}"
        )

        print(
            f"Length     : {len(chunk.text)}"
        )

        print(
            f"Annotations: {chunk.annotation_count}"
        )

        print("\nTEXT:")
        print(chunk.text)

    # -----------------------------------------------------------------
    # Debug export
    # -----------------------------------------------------------------

    debug_path = (
        OUTPUT_DIR /
        f"{document.document_id}_chunk_debug.json"
    )

    export_chunk_debug(
        chunks,
        debug_path,
    )

    print("\n")
    print("=" * 80)
    print(f"Chunk debug written to:")
    print(debug_path)
    print("=" * 80)