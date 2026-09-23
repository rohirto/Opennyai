from pathlib import Path
import sys
import json
import time

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# OpenNyAI dependencies
# ---------------------------------------------------------------------

import sentencepiece
import transformers
import spacy
import spacy_transformers

from opennyai import Pipeline
from legal_preprocessing.judgment_ingester import ingest_legal_document


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PDF_PATH = PROJECT_ROOT / "input" / "doc.pdf"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / f"{PDF_PATH.stem}_ner_result.json"
)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("OpenNyAI Legal NER")
    print("=" * 70)

    print(f"\nInput PDF : {PDF_PATH}")
    print(f"Output    : {OUTPUT_PATH}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # ---------------------------------------------------------------
    # 1. Ingest and preprocess PDF
    # ---------------------------------------------------------------

    print("\n[1/5] Ingesting PDF...")

    start = time.perf_counter()

    document = ingest_legal_document(PDF_PATH)

    ingestion_time = time.perf_counter() - start

    print(
        f"      Ingestion completed in "
        f"{ingestion_time:.2f} seconds"
    )

    print(
        f"      Raw characters     : "
        f"{len(document.raw_text):,}"
    )

    print(
        f"      Cleaned characters : "
        f"{len(document.cleaned_text):,}"
    )

    # ---------------------------------------------------------------
    # 2. Validate extracted text
    # ---------------------------------------------------------------

    if not document.cleaned_text.strip():
        raise RuntimeError(
            "Legal document contains no usable text after preprocessing."
        )

    # ---------------------------------------------------------------
    # 3. Create spaCy document
    # ---------------------------------------------------------------

    print("\n[2/5] Creating spaCy document...")

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    judgement_doc = nlp(
        document.cleaned_text
    )

    print(
        f"      Characters : {len(document.cleaned_text):,}"
    )

    print(
        f"      Tokens     : {len(judgement_doc):,}"
    )

    print(
        f"      Sentences  : "
        f"{len(list(judgement_doc.sents)):,}"
    )

    # ---------------------------------------------------------------
    # 4. Prepare OpenNyAI input
    # ---------------------------------------------------------------

    data = [
        {
            "judgement_doc": judgement_doc,
            "preamble_doc": nlp(""),
            "file_id": PDF_PATH.stem,
            "original_text": document.cleaned_text,
        }
    ]

    # ---------------------------------------------------------------
    # 5. Load and run NER
    # ---------------------------------------------------------------

    print("\n[3/5] Loading OpenNyAI NER...")

    start = time.perf_counter()

    pipeline = Pipeline(
        components=["NER"],
        use_gpu=False,
        verbose=True,
    )

    ner_load_time = time.perf_counter() - start

    print(
        f"\n      NER loaded in "
        f"{ner_load_time:.2f} seconds"
    )

    print("\n[4/5] Running NER...")

    start = time.perf_counter()

    result = pipeline(data)

    ner_time = time.perf_counter() - start

    print(
        f"\n      NER completed in "
        f"{ner_time:.2f} seconds"
    )

    # ---------------------------------------------------------------
    # Save result
    # ---------------------------------------------------------------

    print("\n[5/5] Saving result...")

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"      Saved to: {OUTPUT_PATH}"
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    entities = []

    for document_result in result:

        for annotation in document_result.get(
            "annotations",
            [],
        ):

            entities.extend(
                annotation.get(
                    "entities",
                    [],
                )
            )

    label_counts = {}

    for entity in entities:

        for label in entity.get(
            "labels",
            [],
        ):

            label_counts[label] = (
                label_counts.get(label, 0) + 1
            )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("NER SUMMARY")
    print("=" * 70)

    print(
        f"Document       : {PDF_PATH.name}"
    )

    print(
        f"Total entities : {len(entities):,}"
    )

    print("\nEntities by label:")

    for label, count in sorted(
        label_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):

        print(
            f"  {label:<30} {count:>6}"
        )

    print("\nSample entities:")

    for entity in entities[:50]:

        labels = ", ".join(
            entity.get("labels", [])
        )

        print(
            f"  [{labels}] "
            f"{entity.get('text')!r}"
        )

    print("\n" + "=" * 70)
    print("TIMING")
    print("=" * 70)

    print(
        f"Ingestion      : {ingestion_time:.2f} sec"
    )

    print(
        f"NER loading    : {ner_load_time:.2f} sec"
    )

    print(
        f"NER inference  : {ner_time:.2f} sec"
    )

    print(
        f"Total          : "
        f"{ingestion_time + ner_load_time + ner_time:.2f} sec"
    )


if __name__ == "__main__":
    main()