import json
import sys
import time
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# IMPORTANT: preload transformer dependencies
# ---------------------------------------------------------

import sentencepiece
import transformers
import spacy_transformers

from opennyai import ExtractiveSummarizer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RRC_FILE = (
    PROJECT_ROOT
    / "output"
    / "rhetorical_role_sunil_kumar_result.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "output"
    / "summarizer_sunil_kumar_result.json"
)

OUTPUT_TXT = (
    PROJECT_ROOT
    / "output"
    / "summarizer_sunil_kumar_summary.txt"
)

RRC_OUTPUT = PROJECT_ROOT/ "output" / "rhetorical_role_sunil_kumar_result.json"

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("OpenNyAI Extractive Summarizer - Sunil Kumar Judgment")
    print("=" * 80)

    # -----------------------------------------------------
    # Load RRC output
    # -----------------------------------------------------

    print("\n[1] Loading Rhetorical Role output...")

    start = time.perf_counter()

    with open(
        RRC_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        rrc_data = json.load(f)

    load_time = time.perf_counter() - start

    print(f"    File: {RRC_FILE}")
    print(f"    Load time: {load_time:.2f}s")

    # -----------------------------------------------------
    # Validate structure
    # -----------------------------------------------------

    print("\n[2] Validating RRC structure...")

    with open(RRC_OUTPUT, "r", encoding="utf-8") as f:
        rrc_data = json.load(f)

    # OpenNyAI RRC returns a list of document objects.
    if isinstance(rrc_data, list):
        if len(rrc_data) != 1:
            raise ValueError(
                f"Expected exactly one RRC document, found {len(rrc_data)}."
            )
        rrc_data = rrc_data[0]

    if not isinstance(rrc_data, dict):
        raise TypeError(
            f"Unexpected RRC output type: {type(rrc_data).__name__}"
        )

    required_keys = {"id", "data", "annotations"}
    missing = required_keys - set(rrc_data.keys())

    if missing:
        raise ValueError(
            f"RRC output is missing required keys: {sorted(missing)}"
        )

    if not isinstance(rrc_data["data"], dict):
        raise TypeError("RRC 'data' must be a JSON object.")

    if "text" not in rrc_data["data"]:
        raise ValueError("RRC 'data' does not contain 'text'.")

    if not isinstance(rrc_data["annotations"], list):
        raise TypeError("RRC 'annotations' must be a list.")

    print(f"    Document ID: {rrc_data['id']}")
    print(f"    Text chars: {len(rrc_data['data']['text']):,}")
    print(f"    Annotations: {len(rrc_data['annotations']):,}")

    # -----------------------------------------------------
    # Show RRC distribution
    # -----------------------------------------------------

    role_counter = Counter()
    document_text = rrc_data["data"]["text"]
    annotations = rrc_data["annotations"]
    
    for annotation in annotations:

        labels = annotation.get("labels", [])

        for label in labels:
            role_counter[label] += 1

    print("\n    Rhetorical role distribution:")

    for role, count in role_counter.most_common():
        print(f"      {role:<20} {count:>5}")

    # -----------------------------------------------------
    # Prepare summarizer input
    # -----------------------------------------------------

    summarizer_input = {
        "id": rrc_data["id"],
        "annotations": rrc_data["annotations"],
        "data": rrc_data["data"],
    }

    # Preserve metadata if present
    if "meta" in rrc_data:
        summarizer_input["meta"] = rrc_data["meta"]

    # -----------------------------------------------------
    # Load summarizer
    # -----------------------------------------------------

    print("\n[3] Loading Extractive Summarizer...")

    start = time.perf_counter()

    summarizer = ExtractiveSummarizer(
        use_gpu=False,
        verbose=True,
        summary_length=0.0,
    )

    model_load_time = time.perf_counter() - start

    print(
        f"\n    Summarizer load time: "
        f"{model_load_time:.2f}s"
    )

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------

    print("\n[4] Running extractive summarization...")

    start = time.perf_counter()

    result = summarizer(
        [summarizer_input]
    )

    inference_time = time.perf_counter() - start

    print(
        f"\n    Inference time: "
        f"{inference_time:.2f}s"
    )

    # -----------------------------------------------------
    # Inspect result
    # -----------------------------------------------------

    print("\n[5] Inspecting output...")

    print(
        f"    Result type: {type(result).__name__}"
    )

    print(
        f"    Result items: {len(result)}"
    )

    if not result:
        raise RuntimeError(
            "Summarizer returned an empty result."
        )

    first_result = result[0]

    print(
        f"    Result keys: "
        f"{list(first_result.keys())}"
    )

    summaries = first_result.get(
        "summaries",
        []
    )

    print(
        f"    Summary objects: "
        f"{len(summaries)}"
    )

    # -----------------------------------------------------
    # Save raw JSON
    # -----------------------------------------------------

    output_payload = {
        "source_rrc_file": str(RRC_FILE),
        "document_id": rrc_data.get("id"),
        "document_characters": len(document_text),
        "annotation_count": len(annotations),
        "rhetorical_role_distribution": dict(
            role_counter
        ),
        "timings": {
            "rrc_load_seconds": round(
                load_time,
                3,
            ),
            "model_load_seconds": round(
                model_load_time,
                3,
            ),
            "inference_seconds": round(
                inference_time,
                3,
            ),
        },
        "summarizer_result": result,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n    Raw result saved to:"
        f"\n    {OUTPUT_JSON}"
    )

    # -----------------------------------------------------
    # Create human-readable output
    # -----------------------------------------------------

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "OpenNyAI Extractive Summary\n"
        )
        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Document: {rrc_data.get('id')}\n"
        )
        f.write(
            f"Source characters: {len(document_text):,}\n"
        )
        f.write(
            f"RRC annotations: {len(annotations):,}\n\n"
        )

        f.write(
            "Rhetorical Role Distribution\n"
        )
        f.write(
            "-" * 80
            + "\n"
        )

        for role, count in role_counter.most_common():
            f.write(
                f"{role:<20} {count:>5}\n"
            )

        f.write("\n\n")

        f.write(
            "SUMMARY OUTPUT\n"
        )
        f.write(
            "=" * 80
            + "\n\n"
        )

        # -------------------------------------------------
        # Dump summary objects in a readable form.
        # -------------------------------------------------

        def write_value(value, indent=0):

            prefix = " " * indent

            if isinstance(value, dict):

                for key, val in value.items():

                    f.write(
                        f"{prefix}{key}:\n"
                    )

                    write_value(
                        val,
                        indent + 2,
                    )

            elif isinstance(value, list):

                for index, item in enumerate(value):

                    f.write(
                        f"{prefix}[{index}]\n"
                    )

                    write_value(
                        item,
                        indent + 2,
                    )

            else:

                f.write(
                    f"{prefix}{value}\n"
                )

        write_value(summaries)

    print(
        f"    Human-readable summary saved to:"
        f"\n    {OUTPUT_TXT}"
    )

    # -----------------------------------------------------
    # Final timing
    # -----------------------------------------------------

    total_time = (
        load_time
        + model_load_time
        + inference_time
    )

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)

    print(
        f"RRC load:       {load_time:.2f}s"
    )
    print(
        f"Model loading:  {model_load_time:.2f}s"
    )
    print(
        f"Inference:      {inference_time:.2f}s"
    )
    print(
        f"Total:          {total_time:.2f}s"
    )

    print("\nOutputs:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_TXT}")


if __name__ == "__main__":
    main()