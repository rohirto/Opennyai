# scripts/test_rhetorical_role.py
from pathlib import Path
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# IMPORTANT:
# These imports must happen BEFORE importing/initializing OpenNyAI Data.
# They prevent the Windows native access violation encountered during
# spaCy/plugin initialization in this legacy environment.
# ---------------------------------------------------------------------------
import sentencepiece
import transformers
import spacy_transformers

import pdfplumber

from opennyai import Pipeline
from opennyai.utils import Data
from legal_preprocessing.judgment_ingester import (
    ingest_legal_document,
)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Input PDF
PDF_PATH = PROJECT_ROOT / "input" / "doc.pdf"

# Output JSON
OUTPUT_PATH = PROJECT_ROOT / "output" / "rhetorical_role_sunil_kumar_result.json"

# OpenNyAI preprocessing model compatible with spaCy 3.2.6
PREPROCESSING_MODEL = "en_core_web_sm"

# GTX 1650 / current legacy environment
USE_GPU = False

# Number of sample predictions to display
NUM_SAMPLE_PREDICTIONS = 30


# ===========================================================================
# PDF EXTRACTION
# ===========================================================================

def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from the PDF using pdfplumber.

    Returns:
        str: Complete extracted text.
    """

    print("=" * 80)
    print("PDF EXTRACTION")
    print("=" * 80)

    print(f"Reading: {pdf_path}")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found:\n{pdf_path}"
        )

    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:

        print(f"Pages found: {len(pdf.pages)}")

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text()

            if text:
                page_texts.append(text)

            if page_number % 10 == 0 or page_number == len(pdf.pages):
                print(
                    f"  Extracted {page_number}/{len(pdf.pages)} pages"
                )

    full_text = "\n\n".join(page_texts)

    print()
    print(f"Pages extracted : {len(page_texts)}")
    print(f"Characters      : {len(full_text):,}")

    return full_text


# ===========================================================================
# PREAMBLE / JUDGMENT SPLIT
# ===========================================================================

def split_preamble_and_judgment(text: str):
    """
    Split the extracted Supreme Court judgment into preamble and judgment.

    This is intentionally kept simple because OpenNyAI's Data object performs
    its own preprocessing and judgment splitting.

    The split is primarily for diagnostics and reporting.
    """

    # Common marker in the supplied Supreme Court judgment.
    marker = "J U D G M E N T"

    index = text.find(marker)

    if index == -1:
        print(
            "\nWARNING: Could not find 'J U D G M E N T' marker."
        )
        print(
            "The complete extracted text will be passed to OpenNyAI Data."
        )

        return "", text

    preamble = text[:index]
    judgment = text[index:]

    return preamble, judgment


# ===========================================================================
# RUN RHETORICAL ROLE MODEL
# ===========================================================================

def run_rhetorical_role(text: str):

    print()
    print("=" * 80)
    print("OPENNYAI DATA PREPROCESSING")
    print("=" * 80)

    print(f"Preprocessing model : {PREPROCESSING_MODEL}")
    print(f"GPU                 : {USE_GPU}")

    start_time = time.perf_counter()

    data = Data(
        text,
        preprocessing_nlp_model=PREPROCESSING_MODEL,
        use_gpu=USE_GPU,
        verbose=True,
    )

    data_time = time.perf_counter() - start_time

    print()
    print(f"Data object created in {data_time:.2f} seconds")

    # -----------------------------------------------------------------------
    # Rhetorical Role Pipeline
    # -----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING RHETORICAL ROLE PIPELINE")
    print("=" * 80)

    pipeline_start = time.perf_counter()

    pipeline = Pipeline(
        components=["Rhetorical_Role"],
        use_gpu=USE_GPU,
        verbose=True,
    )

    pipeline_load_time = time.perf_counter() - pipeline_start

    print()
    print(
        f"Pipeline loaded in {pipeline_load_time:.2f} seconds"
    )

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING RHETORICAL ROLE INFERENCE")
    print("=" * 80)

    inference_start = time.perf_counter()

    result = pipeline(data)

    inference_time = time.perf_counter() - inference_start

    print()
    print(
        f"Inference completed in {inference_time:.2f} seconds"
    )

    return result, data_time, pipeline_load_time, inference_time


# ===========================================================================
# ANALYZE RESULT
# ===========================================================================

def analyze_result(result):

    print()
    print("=" * 80)
    print("RESULT ANALYSIS")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Basic structure
    # -----------------------------------------------------------------------

    if not result:
        print("ERROR: OpenNyAI returned an empty result.")
        return

    print(f"Documents returned: {len(result)}")

    total_annotations = 0

    all_annotations = []

    for document_index, document in enumerate(result, start=1):

        document_id = document.get("id")

        data = document.get("data", {})

        annotations = document.get("annotations", [])

        total_annotations += len(annotations)

        all_annotations.extend(annotations)

        print()
        print(f"Document {document_index}")
        print("-" * 80)

        print(f"ID                    : {document_id}")
        print(
            f"Text characters       : "
            f"{len(data.get('text', '')):,}"
        )
        print(
            f"Preamble end offset   : "
            f"{data.get('preamble_end_char_offset')}"
        )
        print(
            f"Annotations            : "
            f"{len(annotations)}"
        )

    print()
    print(f"Total annotations: {total_annotations}")

    # -----------------------------------------------------------------------
    # Role distribution
    # -----------------------------------------------------------------------

    role_counter = Counter()

    for annotation in all_annotations:

        labels = annotation.get("labels", [])

        if labels:
            role_counter[labels[0]] += 1

    print()
    print("=" * 80)
    print("RHETORICAL ROLE DISTRIBUTION")
    print("=" * 80)

    if not role_counter:
        print("No rhetorical role labels were returned.")
    else:

        for role, count in role_counter.most_common():

            percentage = (
                count / total_annotations * 100
                if total_annotations
                else 0
            )

            print(
                f"{role:20s} "
                f"{count:6d} "
                f"({percentage:6.2f}%)"
            )

    # -----------------------------------------------------------------------
    # Sample predictions
    # -----------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"FIRST {min(NUM_SAMPLE_PREDICTIONS, len(all_annotations))} "
        "RHETORICAL ROLE PREDICTIONS"
    )
    print("=" * 80)

    for index, annotation in enumerate(
        all_annotations[:NUM_SAMPLE_PREDICTIONS],
        start=1,
    ):

        labels = annotation.get("labels", [])

        role = labels[0] if labels else "UNKNOWN"

        sentence = annotation.get("text", "")

        print()
        print(
            f"{index:02d}. [{role:15s}]"
        )
        print(
            f"    {sentence}"
        )

    return role_counter


# ===========================================================================
# SAVE RESULT
# ===========================================================================

def save_result(result, output_path: Path):

    print()
    print("=" * 80)
    print("SAVING RESULT")
    print("=" * 80)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Result saved to:")
    print(output_path)

    print(
        f"File size: {output_path.stat().st_size / 1024:.1f} KB"
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    total_start = time.perf_counter()

    print()
    print("=" * 80)
    print("OPENNYAI — RHETORICAL ROLE TEST")
    print("=" * 80)

    print()
    print("Environment")
    print("-" * 80)

    print(f"PDF                 : {PDF_PATH}")
    print(f"Preprocessing model : {PREPROCESSING_MODEL}")
    print(f"GPU                 : {USE_GPU}")
    print(f"Output              : {OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # 1. Extract PDF
    # -----------------------------------------------------------------------

    document = ingest_legal_document(PDF_PATH)

    raw_text = document.raw_text
    cleaned_text = document.cleaned_text
    report = document.preprocessing_report
    # -----------------------------------------------------------------------
    # 2. Diagnostic preamble/judgment split
    # -----------------------------------------------------------------------

    preamble, judgment = split_preamble_and_judgment(cleaned_text)

    print()
    print("=" * 80)
    print("TEXT SUMMARY")
    print("=" * 80)

    print(
        f"Total characters     : {len(cleaned_text):,}"
    )

    print(
        f"Preamble characters  : {len(preamble):,}"
    )

    print(
        f"Judgment characters  : {len(judgment):,}"
    )

    # -----------------------------------------------------------------------
    # 3. OpenNyAI Data + Rhetorical Role inference
    # -----------------------------------------------------------------------

    result, data_time, pipeline_time, inference_time = (
        run_rhetorical_role(cleaned_text)
    )

    # -----------------------------------------------------------------------
    # 4. Analyze result
    # -----------------------------------------------------------------------

    role_counter = analyze_result(result)

    # -----------------------------------------------------------------------
    # 5. Save JSON
    # -----------------------------------------------------------------------

    save_result(
        result,
        OUTPUT_PATH,
    )

    # -----------------------------------------------------------------------
    # 6. Final summary
    # -----------------------------------------------------------------------

    total_time = time.perf_counter() - total_start

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"PDF characters       : {len(document.cleaned_text):,}"
    )

    print(
        f"Preamble characters  : {len(preamble):,}"
    )

    print(
        f"Judgment characters  : {len(judgment):,}"
    )

    print(
        f"Data preprocessing   : {data_time:.2f} sec"
    )

    print(
        f"Pipeline loading     : {pipeline_time:.2f} sec"
    )

    print(
        f"Rhetorical inference : {inference_time:.2f} sec"
    )

    print(
        f"Total execution      : {total_time:.2f} sec"
    )

    if role_counter:
        print()
        print("Roles detected:")
        for role, count in role_counter.most_common():
            print(f"  {role:20s}: {count}")

    print()
    print("STATUS: Rhetorical Role pipeline completed successfully.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()