from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INPUT_DIR, OUTPUT_DIR, ensure_directories
from legal_preprocessing.judgment_ingester import ingest_legal_document
from legal_preprocessing.ner_runner import NERRunner
from legal_preprocessing.rrc_runner import RRCRunner
from legal_preprocessing.summarizer_runner import SummarizerRunner
from legal_preprocessing.summary_formatter import save_summary


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def process_judgment(
    pdf_path: Path, 
    ner_runner: NERRunner, 
    rrc_runner: RRCRunner,
    summarizer_runner: SummarizerRunner
    ):

    document_id = pdf_path.stem

    output_dir = OUTPUT_DIR / document_id
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    
    
    
    print("=" * 80)
    print(f"PROCESSING: {pdf_path.name}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. INGESTION
    # ------------------------------------------------------------------

    print("\n[1/4] Ingesting document...")

    document = ingest_legal_document(pdf_path)

    print(
        f"Raw characters     : {len(document.raw_text):,}"
    )

    print(
        f"Cleaned characters : {len(document.cleaned_text):,}"
    )

    # Save cleaned text
    cleaned_text_path = output_dir / "cleaned_text.txt"

    cleaned_text_path.write_text(
        document.cleaned_text,
        encoding="utf-8",
    )

    # Save preprocessing report
    save_json(
        output_dir / "preprocessing_report.json",
        document.preprocessing_report,
    )


    # ------------------------------------------------------------------
    # 2. NER
    # ------------------------------------------------------------------

    print("\n[2/4] Running NER...")

    ner_result = ner_runner.run(document)

    save_json(
        output_dir / "ner.json",
        ner_result,
    )
    entities = []

    for output_document in ner_result:
        for annotation in output_document.get(
            "annotations",
            []
        ):
            entities.extend(
                annotation.get(
                    "entities",
                    []
                )
            )
    from collections import Counter
    label_counts = Counter()

    for entity in entities:
        for label in entity.get(
            "labels",
            []
        ):
            label_counts[label] += 1
    
    print(
        f"Total entities: {len(entities):,}"
    )

    for label, count in label_counts.most_common():
        print(
            f"  {label:<25} {count:>6}"
        )
    

    # ------------------------------------------------------------------
    # 3. RHETORICAL ROLE CLASSIFICATION
    # ------------------------------------------------------------------

    print(
        "\n[3/4] Running rhetorical role classification..."
    )

    rrc_result = rrc_runner.run(document)

    save_json(
        output_dir / "rhetorical_roles.json",
        rrc_result,
    )

    # ------------------------------------------------------------------
    # 4. SUMMARIZATION
    # ------------------------------------------------------------------

    print("\n[4/4] Running extractive summarizer...")


    summarizer_result = summarizer_runner.run(rrc_result)

    save_json(
        output_dir / "summarizer.json",
        summarizer_result,
    )
    
    summary_text = save_summary(
        output_dir / "summary.txt",
        summarizer_result,
    )

    print(f"\nCompleted: {document_id}")


def main():

    ensure_directories()

    pdf_files = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".pdf"}
    )

    if not pdf_files:
        print(
            f"No PDF files found in: {INPUT_DIR}"
        )
        return

    print(
        f"Found {len(pdf_files)} PDF(s)"
    )
    
    # Load ONCE
    ner_runner = NERRunner(
        use_gpu=True,
        verbose=True,
    )
    rrc_runner = RRCRunner(
        use_gpu=True,
        verbose=True,
    )
    summarizer_runner = SummarizerRunner(
        use_gpu=True,
        verbose=True,
    )

    for pdf_path in pdf_files:
        process_judgment(pdf_path, ner_runner, rrc_runner,summarizer_runner)


if __name__ == "__main__":
    main()