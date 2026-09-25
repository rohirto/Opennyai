"""
Legacy/OpenNyAI ingestion stage.

RUN THIS SCRIPT WITH:
    .venv-legacy

Responsibilities
----------------
PDF
  -> pdfplumber extraction
  -> LegalTextCleaner
  -> OpenNyAI NER
  -> OpenNyAI RRC
  -> OpenNyAI Extractive Summarizer
  -> serialized artifacts for venv-rag

This script MUST NOT import anything from the RAG pipeline.

Output
------
output/
└── <document_name>/
    ├── cleaned_text.txt
    ├── preprocessing_report.json
    ├── ner.json
    ├── rhetorical_roles.json
    ├── summary_raw.json
    └── summary.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CUDA bootstrap
# IMPORTANT: PyTorch must be imported before CuPy / spaCy / Thinc / OpenNyAI
# on this Windows environment.
# ---------------------------------------------------------------------------

import torch

if torch.cuda.is_available():
    # Force CUDA runtime DLLs to be loaded into the process.
    torch.cuda.init()





# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ============================================================================
# CONFIGURATION
# ============================================================================

USE_GPU = True
VERBOSE = True

# Process only PDFs.
PDF_PATTERN = "*.pdf"


# ============================================================================
# IMPORTS
# ============================================================================

# The project root must be on sys.path when this script is launched directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from legal_preprocessing.judgment_ingester import (
    ingest_legal_document,
)

from legal_preprocessing.ner_runner import (
    NERRunner,
)

from legal_preprocessing.rrc_runner import (
    RRCRunner,
)

from legal_preprocessing.summarizer_runner import (
    SummarizerRunner,
)

from legal_preprocessing.summary_formatter import (
    save_summary,
)


# ============================================================================
# JSON HELPERS
# ============================================================================

def save_json(
    data: Any,
    path: Path,
) -> None:
    """Save JSON using UTF-8 and preserve Unicode."""

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
# DOCUMENT PROCESSING
# ============================================================================

def process_document(
    pdf_path: Path,
    ner_runner: NERRunner,
    rrc_runner: RRCRunner,
    summarizer_runner: SummarizerRunner,
) -> bool:

    start_time = time.perf_counter()

    document_name = pdf_path.stem

    output_dir = (
        OUTPUT_DIR
        / document_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print(f"PROCESSING: {pdf_path.name}")
    print("=" * 90)

    # ------------------------------------------------------------------------
    # 1. PDF extraction + legal text cleaning
    # ------------------------------------------------------------------------

    print()
    print("[1/4] Extracting and cleaning PDF...")

    document = ingest_legal_document(
        pdf_path
    )

    print(
        f"Raw characters     : "
        f"{len(document.raw_text):,}"
    )

    print(
        f"Cleaned characters : "
        f"{len(document.cleaned_text):,}"
    )

    # Save authoritative cleaned text.
    cleaned_text_path = (
        output_dir
        / "cleaned_text.txt"
    )

    cleaned_text_path.write_text(
        document.cleaned_text,
        encoding="utf-8",
    )

    # Save preprocessing report.
    save_json(
        document.preprocessing_report,
        output_dir / "preprocessing_report.json",
    )

    # ------------------------------------------------------------------------
    # 2. NER
    # ------------------------------------------------------------------------

    print()
    print("[2/4] Running NER...")

    ner_result = ner_runner.run(
        document
    )

    save_json(
        ner_result,
        output_dir / "ner.json",
    )

    # Count entities defensively.
    entity_count = 0

    if isinstance(ner_result, list):

        for output_document in ner_result:

            if not isinstance(
                output_document,
                dict,
            ):
                continue

            for annotation in output_document.get(
                "annotations",
                [],
            ):

                if isinstance(
                    annotation,
                    dict,
                ):

                    entity_count += len(
                        annotation.get(
                            "entities",
                            [],
                        )
                    )

    print(
        f"NER entities      : "
        f"{entity_count:,}"
    )

    # ------------------------------------------------------------------------
    # 3. Rhetorical Role Classification
    # ------------------------------------------------------------------------

    print()
    print("[3/4] Running rhetorical role classification...")

    rrc_result = rrc_runner.run(
        document
    )

    save_json(
        rrc_result,
        output_dir / "rhetorical_roles.json",
    )

    # Diagnostics.
    annotation_count = 0
    role_counts: dict[str, int] = {}

    if isinstance(rrc_result, list):

        for output_document in rrc_result:

            if not isinstance(
                output_document,
                dict,
            ):
                continue

            annotations = output_document.get(
                "annotations",
                [],
            )

            annotation_count += len(
                annotations
            )

            for annotation in annotations:

                if not isinstance(
                    annotation,
                    dict,
                ):
                    continue

                labels = annotation.get(
                    "labels",
                    [],
                )

                if isinstance(
                    labels,
                    str,
                ):
                    role = labels.strip()

                elif labels:
                    role = str(
                        labels[0]
                    ).strip()

                else:
                    role = "UNKNOWN"

                role_counts[role] = (
                    role_counts.get(
                        role,
                        0,
                    )
                    + 1
                )

    print(
        f"RRC annotations   : "
        f"{annotation_count:,}"
    )

    if role_counts:

        print()
        print("Rhetorical roles:")

        for role, count in sorted(
            role_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            print(
                f"  {role:20s}: {count:,}"
            )

    # ------------------------------------------------------------------------
    # 4. Extractive summarization
    # ------------------------------------------------------------------------

    print()
    print("[4/4] Running extractive summarizer...")

    # IMPORTANT:
    #
    # SummarizerRunner.run() expects the RRC result,
    # not the preprocessing LegalDocument.
    #
    summary_result = summarizer_runner.run(
        rrc_result
    )

    save_json(
        summary_result,
        output_dir / "summary_raw.json",
    )

    # Human-readable summary.
    summary_text = save_summary(
        output_dir / "summary.txt",
        summary_result,
    )

    # ------------------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("-" * 90)
    print("LEGACY INGESTION COMPLETE")
    print("-" * 90)

    print(
        f"Document           : "
        f"{document_name}"
    )

    print(
        f"Cleaned characters : "
        f"{len(document.cleaned_text):,}"
    )

    print(
        f"NER entities       : "
        f"{entity_count:,}"
    )

    print(
        f"RRC annotations    : "
        f"{annotation_count:,}"
    )

    print(
        f"Summary characters : "
        f"{len(summary_text):,}"
    )

    print(
        f"Elapsed time       : "
        f"{elapsed:.2f} sec"
    )

    print()
    print("Artifacts:")
    print(
        f"  {output_dir / 'cleaned_text.txt'}"
    )
    print(
        f"  {output_dir / 'preprocessing_report.json'}"
    )
    print(
        f"  {output_dir / 'ner.json'}"
    )
    print(
        f"  {output_dir / 'rhetorical_roles.json'}"
    )
    print(
        f"  {output_dir / 'summary_raw.json'}"
    )
    print(
        f"  {output_dir / 'summary.txt'}"
    )

    return True

def verify_cuda_stack():
    import torch

    print("=" * 70)
    print("CUDA STACK CHECK")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")

    print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------------------------------------------------
    # CuPy
    # -----------------------------------------------------------------------

    import cupy

    print("CuPy:", cupy.__version__)
    print("CuPy devices:", cupy.cuda.runtime.getDeviceCount())
    print("CuPy runtime:", cupy.cuda.runtime.runtimeGetVersion())

    x = cupy.arange(10)
    print("CuPy allocation:", x)

    # -----------------------------------------------------------------------
    # Torch -> CuPy DLPack
    # -----------------------------------------------------------------------

    torch_tensor = torch.arange(
        10,
        device="cuda",
    )

    cupy_tensor = cupy.from_dlpack(torch_tensor)

    print("Torch -> CuPy DLPack:", cupy_tensor)

    # -----------------------------------------------------------------------
    # Thinc
    # -----------------------------------------------------------------------

    import thinc.util

    print("Thinc has_cupy:", thinc.util.has_cupy)

    if not thinc.util.has_cupy:
        raise RuntimeError(
            "Thinc cannot see CuPy even though CuPy itself is working."
        )

    thinc.util.require_gpu()

    print("Thinc GPU: OK")
    print("=" * 70)
# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print()
    print("=" * 90)
    print("OPENNYAI LEGACY INGESTION")
    print("=" * 90)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input        : {INPUT_DIR}")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"GPU          : {USE_GPU}")

    if not INPUT_DIR.exists():

        print()
        print(
            f"ERROR: Input directory does not exist: "
            f"{INPUT_DIR}"
        )

        return 1

    pdf_files = sorted(
        INPUT_DIR.glob(PDF_PATTERN)
    )

    if not pdf_files:

        print()
        print(
            f"No PDF files found in: "
            f"{INPUT_DIR}"
        )

        return 0

    print()
    print(
        f"Found {len(pdf_files)} PDF(s)"
    )

    # ------------------------------------------------------------------------
    # Load OpenNyAI models ONCE.
    #
    # This is important because loading NER/RRC/Summarizer for every PDF
    # would be unnecessarily expensive.
    # ------------------------------------------------------------------------
    verify_cuda_stack()
    print()
    print("=" * 90)
    print("INITIALIZING OPENNYAI RUNNERS")
    print("=" * 90)

    ner_runner = NERRunner(
        use_gpu=USE_GPU,
        verbose=VERBOSE,
    )

    rrc_runner = RRCRunner(
        use_gpu=USE_GPU,
        verbose=VERBOSE,
    )

    summarizer_runner = SummarizerRunner(
        use_gpu=USE_GPU,
        verbose=VERBOSE,
    )

    successful = 0
    failed = 0

    # ------------------------------------------------------------------------
    # Process PDFs independently.
    # ------------------------------------------------------------------------

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(pdf_files)}]"
        )

        try:

            success = process_document(
                pdf_path=pdf_path,
                ner_runner=ner_runner,
                rrc_runner=rrc_runner,
                summarizer_runner=summarizer_runner,
            )

            if success:
                successful += 1
            else:
                failed += 1

        except Exception as exc:

            failed += 1

            print()
            print("!" * 90)
            print(
                f"FAILED: {pdf_path.name}"
            )
            print("!" * 90)

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            # Continue with the remaining PDFs.
            continue

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("LEGACY INGESTION SUMMARY")
    print("=" * 90)

    print(
        f"Total PDFs : {len(pdf_files)}"
    )

    print(
        f"Successful : {successful}"
    )

    print(
        f"Failed     : {failed}"
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