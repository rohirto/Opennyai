"""
Inspect specific chunks from a generated chunks.json file.

Usage:

    python scripts/inspect_chunks.py

By default this inspects the chunks that appeared in the
end-to-end retrieval test.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOCUMENT = "Sunil_Kumar_B"

CHUNK_INDICES = [54, 60, 62, 101, 106]

CHUNKS_FILE = ROOT / "output" / DOCUMENT / "chunks.json"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.json not found:\n{CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", chunks)

    by_index = {
        int(chunk["chunk_index"]): chunk
        for chunk in chunks
    }

    print("=" * 100)
    print("CHUNK INSPECTION")
    print("=" * 100)

    print(f"Document : {DOCUMENT}")
    print(f"File     : {CHUNKS_FILE}")
    print(f"Chunks   : {len(chunks)}")

    for index in CHUNK_INDICES:

        chunk = by_index.get(index)

        print()
        print("=" * 100)
        print(f"CHUNK {index}")
        print("=" * 100)

        if chunk is None:
            print("NOT FOUND")
            continue

        fields = [
            "chunk_index",
            "text",
            "embedding_text",
            "primary_role",
            "roles",
            "role_sequence",
            "annotation_ids",
            "char_start",
            "char_end",
            "source_char_count",
            "text_char_count",
            "annotation_count",
            "entities",
            "entity_ids",
            "summary_context",
            "document_id",
            "case_name",
            "citation",
            "court",
            "judgment_date",
        ]

        for field in fields:

            value = chunk.get(field)

            print()
            print(f"--- {field} ---")

            if field in {"text", "embedding_text", "summary_context"}:
                print(value)
            else:
                print(repr(value))


if __name__ == "__main__":
    main()
