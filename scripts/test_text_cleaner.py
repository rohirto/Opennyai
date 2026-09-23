"""
Diagnostic test script for LegalTextCleaner V2.

This script:
    1. Extracts text from the PDF
    2. Runs LegalTextCleaner V2
    3. Displays before/after extraction artifacts
    4. Displays the cleaning report
    5. Displays suspicious joined-word candidates
    6. Displays context for suspicious candidates
    7. Saves cleaned text
    8. Saves JSON cleaning report
    9. Saves a human-readable suspicious-word report

IMPORTANT:
    Suspicious joined words are NOT automatically modified.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pdfplumber


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from legal_preprocessing.legal_text_cleaner import LegalTextCleaner


# =============================================================================
# CONFIGURATION
# =============================================================================

PDF_PATH = PROJECT_ROOT / "input" / "doc.pdf"

OUTPUT_TEXT_PATH = (
    PROJECT_ROOT / "output" / "doc_cleaned_v2.txt"
)

OUTPUT_REPORT_PATH = (
    PROJECT_ROOT / "output" / "text_cleaning_report_v2.json"
)

OUTPUT_SUSPICIOUS_PATH = (
    PROJECT_ROOT / "output" / "suspicious_joined_words_v2.txt"
)

CONTEXT_CHARS = 180


# =============================================================================
# TYPES
# =============================================================================

SuspiciousItem = Union[str, Dict[str, Any]]


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """
    Extract text from all pages of a PDF.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pages_text: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:

        page_count = len(pdf.pages)

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                pages_text.append(text)

    return "\n".join(pages_text), page_count


# =============================================================================
# BEFORE/AFTER DIAGNOSTICS
# =============================================================================

def count_page_markers(text: str) -> int:
    """
    Count common page-marker patterns.
    """

    patterns = [
        r"(?im)^\s*[-–—]?\s*Page\s+\d+\s*[-–—]?\s*$",
        r"(?im)^\s*\d+\s*\|\s*Pag\s*e\s*$",
    ]

    total = 0

    for pattern in patterns:
        total += len(
            re.findall(pattern, text)
        )

    return total


def count_standalone_page_numbers(text: str) -> int:
    """
    Count standalone numeric lines.
    """

    return len(
        re.findall(
            r"(?m)^\s*\d{1,4}\s*$",
            text,
        )
    )


def count_hyphenated_line_breaks(text: str) -> int:
    """
    Count words broken across lines using a hyphen.
    """

    return len(
        re.findall(
            r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])",
            text,
        )
    )


# =============================================================================
# CLEANER RETURN VALUE HANDLING
# =============================================================================

def unpack_cleaner_result(result: Any) -> tuple[str, Any]:
    """
    Unpack the return value from LegalTextCleaner.clean().

    Current V2 cleaner returns a tuple:

        (
            cleaned_text,
            report
        )

    This function deliberately supports a few harmless variations so that
    the diagnostic script does not become unnecessarily brittle.
    """

    # -------------------------------------------------------------------------
    # Expected V2 format
    # -------------------------------------------------------------------------

    if isinstance(result, tuple):

        if len(result) != 2:

            raise TypeError(
                "LegalTextCleaner.clean() returned a tuple with "
                f"{len(result)} elements. Expected 2."
            )

        cleaned_text, report = result

        if not isinstance(cleaned_text, str):

            raise TypeError(
                "First element returned by LegalTextCleaner.clean() "
                f"must be str, got {type(cleaned_text)}"
            )

        return cleaned_text, report

    # -------------------------------------------------------------------------
    # Dictionary format
    # -------------------------------------------------------------------------

    if isinstance(result, dict):

        cleaned_text = result.get(
            "cleaned_text"
        )

        report = result.get(
            "report"
        )

        if not isinstance(cleaned_text, str):

            raise TypeError(
                "Dictionary returned by LegalTextCleaner.clean() does "
                "not contain a string 'cleaned_text'."
            )

        return cleaned_text, report

    # -------------------------------------------------------------------------
    # Object format
    # -------------------------------------------------------------------------

    if hasattr(result, "cleaned_text"):

        cleaned_text = result.cleaned_text

        report = getattr(
            result,
            "report",
            None,
        )

        if not isinstance(cleaned_text, str):

            raise TypeError(
                "LegalTextCleaner result has 'cleaned_text', but it "
                f"is {type(cleaned_text)} instead of str."
            )

        return cleaned_text, report

    raise TypeError(
        "Unexpected return type from "
        "LegalTextCleaner.clean(): "
        f"{type(result)}"
    )


# =============================================================================
# REPORT HELPERS
# =============================================================================

def get_report_value(
    report: Any,
    field: str,
    default: Any = 0,
) -> Any:
    """
    Retrieve a field from either a dataclass/object or dictionary.
    """

    if report is None:
        return default

    if isinstance(report, dict):

        return report.get(
            field,
            default,
        )

    return getattr(
        report,
        field,
        default,
    )


def get_suspicious_items(
    report: Any,
) -> List[SuspiciousItem]:
    """
    Extract suspicious joined-word candidates from the cleaner report.

    IMPORTANT:
    - suspicious_words contains the actual candidate dictionaries.
    - suspicious_joined_words contains only the integer count.
    """

    candidates = get_report_value(
        report,
        "suspicious_words",
        [],
    )

    if candidates is None:
        return []

    if not isinstance(candidates, list):
        return [candidates]

    return candidates
# =============================================================================
# SUSPICIOUS WORD HELPERS
# =============================================================================

def get_suspicious_word(
    item: SuspiciousItem,
) -> str:
    """
    Extract the actual suspicious token.

    Supports both:

        "employeremployee"

    and:

        {
            "word": "employeremployee",
            ...
        }
    """

    if isinstance(item, dict):

        value = item.get(
            "word",
            "",
        )

        return str(value)

    return str(item)


def normalize_suspicious_items(
    items: List[SuspiciousItem],
) -> List[Dict[str, Any]]:
    """
    Normalize suspicious candidates into dictionaries.
    """

    normalized: List[Dict[str, Any]] = []

    for item in items:

        if isinstance(item, dict):

            normalized.append(
                dict(item)
            )

        else:

            normalized.append(
                {
                    "word": str(item)
                }
            )

    return normalized


# =============================================================================
# CONTEXT EXTRACTION
# =============================================================================

def get_word_contexts(
    text: str,
    suspicious_items: List[Dict[str, Any]],
    context_chars: int = CONTEXT_CHARS,
) -> Dict[str, List[str]]:
    """
    Find occurrences of suspicious words and return surrounding context.
    """

    contexts: Dict[str, List[str]] = {}

    for item in suspicious_items:

        word = get_suspicious_word(
            item
        )

        if not word:
            continue

        pattern = re.compile(
            rf"\b{re.escape(word)}\b",
            re.IGNORECASE,
        )

        matches: List[str] = []

        for match in pattern.finditer(text):

            start = max(
                0,
                match.start() - context_chars,
            )

            end = min(
                len(text),
                match.end() + context_chars,
            )

            context = (
                text[start:match.start()]
                + ">>>"
                + text[match.start():match.end()]
                + "<<<"
                + text[match.end():end]
            )

            context = re.sub(
                r"\s+",
                " ",
                context,
            ).strip()

            matches.append(
                context
            )

        contexts[word] = matches

    return contexts


# =============================================================================
# SAVE SUSPICIOUS REPORT
# =============================================================================

def save_suspicious_report(
    output_path: Path,
    suspicious_items: List[Dict[str, Any]],
    contexts: Dict[str, List[str]],
) -> None:

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "LEGAL TEXT CLEANER V2\n"
        )

        f.write(
            "SUSPICIOUS JOINED WORD REVIEW\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Total suspicious candidates: "
            f"{len(suspicious_items)}\n\n"
        )

        if not suspicious_items:

            f.write(
                "No suspicious joined-word candidates detected.\n"
            )

            return

        for index, item in enumerate(
            suspicious_items,
            start=1,
        ):

            word = item.get(
                "word",
                "",
            )

            f.write(
                f"{index}. {word}\n"
            )

            for key, value in item.items():

                if key == "word":
                    continue

                f.write(
                    f"   {key}: {value}\n"
                )

            f.write("\n")

            word_contexts = contexts.get(
                word,
                [],
            )

            f.write(
                "   CONTEXTS:\n"
            )

            if word_contexts:

                for context in word_contexts:

                    f.write(
                        f"   - {context}\n"
                    )

            else:

                f.write(
                    "   - None found\n"
                )

            f.write("\n")

            f.write(
                "-" * 80
                + "\n\n"
            )


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def report_to_dict(
    report: Any,
) -> Dict[str, Any]:
    """
    Convert the cleaner report into a JSON-compatible dictionary.
    """

    if report is None:
        return {}

    if isinstance(report, dict):
        return dict(report)

    result: Dict[str, Any] = {}

    fields = [
        "page_markers_removed",
        "standalone_page_numbers_removed",
        "unicode_normalizations",
        "soft_hyphens_removed",
        "ligatures_normalized",
        "hyphenated_line_breaks_repaired",
        "line_breaks_normalized",
        "suspicious_joined_words",
        "warnings",
    ]

    for field in fields:

        if hasattr(report, field):

            result[field] = getattr(
                report,
                field,
            )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print()
    print("=" * 80)
    print("LEGAL TEXT CLEANER V2")
    print("=" * 80)

    # =========================================================================
    # PDF EXTRACTION
    # =========================================================================

    print(
        f"Reading PDF: {PDF_PATH}"
    )

    raw_text, page_count = extract_pdf_text(
        PDF_PATH
    )

    print(
        f"Pages: {page_count}"
    )

    print(
        f"Raw characters: {len(raw_text):,}"
    )

    # =========================================================================
    # BEFORE CLEANING
    # =========================================================================

    print()
    print("=" * 80)
    print("BEFORE CLEANING")
    print("=" * 80)

    print(
        f"page_marker                   : "
        f"{count_page_markers(raw_text)}"
    )

    print(
        f"page_marker_page              : 0"
    )

    print(
        f"standalone_page_number        : "
        f"{count_standalone_page_numbers(raw_text)}"
    )

    print(
        f"hyphenated_line_break         : "
        f"{count_hyphenated_line_breaks(raw_text)}"
    )

    # =========================================================================
    # RUN CLEANER
    # =========================================================================

    cleaner = LegalTextCleaner()

    cleaner_result = cleaner.clean(
        raw_text
    )

    cleaned_text, report = unpack_cleaner_result(
        cleaner_result
    )

    # =========================================================================
    # SUSPICIOUS WORDS
    # =========================================================================

    suspicious_raw = get_suspicious_items(
        report
    )

    suspicious_items = normalize_suspicious_items(
        suspicious_raw
    )

    # =========================================================================
    # AFTER CLEANING
    # =========================================================================

    print()
    print("=" * 80)
    print("AFTER CLEANING")
    print("=" * 80)

    print(
        f"page_marker                   : "
        f"{count_page_markers(cleaned_text)}"
    )

    print(
        f"page_marker_page              : 0"
    )

    print(
        f"standalone_page_number        : "
        f"{count_standalone_page_numbers(cleaned_text)}"
    )

    print(
        f"hyphenated_line_break         : "
        f"{count_hyphenated_line_breaks(cleaned_text)}"
    )

    # =========================================================================
    # CLEANING REPORT
    # =========================================================================

    print()
    print("=" * 80)
    print("CLEANING REPORT")
    print("=" * 80)

    original_characters = len(raw_text)
    cleaned_characters = len(cleaned_text)

    print(
        f"{'original_characters':40} : "
        f"{original_characters}"
    )

    print(
        f"{'cleaned_characters':40} : "
        f"{cleaned_characters}"
    )

    print(
        f"{'characters_removed':40} : "
        f"{original_characters - cleaned_characters}"
    )

    report_fields = [
        "page_markers_removed",
        "standalone_page_numbers_removed",
        "unicode_normalizations",
        "soft_hyphens_removed",
        "ligatures_normalized",
        "hyphenated_line_breaks_repaired",
        "line_breaks_normalized",
    ]

    for field in report_fields:

        print(
            f"{field:40} : "
            f"{get_report_value(report, field, 0)}"
        )

    print(
        f"{'suspicious_joined_words':40} : "
        f"{len(suspicious_items)}"
    )

    # =========================================================================
    # WARNINGS
    # =========================================================================

    warnings = get_report_value(
        report,
        "warnings",
        [],
    )

    if warnings:

        print()
        print("=" * 80)
        print("WARNINGS")
        print("=" * 80)

        for warning in warnings:

            print(
                f"- {warning}"
            )

    # =============================================================================
    # SUSPICIOUS JOINED WORDS
    # =============================================================================

    suspicious_words = getattr(report, "suspicious_words", [])

    print()
    print("=" * 80)
    print(f"SUSPICIOUS JOINED WORDS ({len(suspicious_words)})")
    print("=" * 80)

    if not suspicious_words:
        print("None detected.")
    else:
        for candidate in suspicious_words:
            print(candidate)


    # =============================================================================
    # SUSPICIOUS WORD CONTEXT
    # =============================================================================

    print()
    print("=" * 80)
    print("SUSPICIOUS WORD CONTEXT")
    print("=" * 80)

    if not suspicious_words:
        print("No suspicious candidates require manual review.")
    else:
        for candidate in suspicious_words:

            word = candidate["word"]

            print()
            print(f"WORD: {word}")
            print("-" * 80)

            # Find occurrences in the cleaned text.
            pattern = re.compile(
                re.escape(word),
                re.IGNORECASE,
            )

            matches = list(pattern.finditer(cleaned_text))

            if not matches:
                print("No occurrence found in cleaned text.")
                continue

            for index, match in enumerate(matches[:20], start=1):

                start = max(0, match.start() - 180)
                end = min(
                    len(cleaned_text),
                    match.end() + 180,
                )

                context = cleaned_text[start:end]

                print(f"[Context {index}]")
                print(
                    context.replace(
                        word,
                        f">>>{word}<<<",
                    )
                )
                print()
    # =========================================================================
    # CONTEXT
    # =========================================================================

    contexts = get_word_contexts(
        cleaned_text,
        suspicious_items,
    )

    if suspicious_items:

        print()
        print("=" * 80)
        print("SUSPICIOUS WORD CONTEXT")
        print("=" * 80)

        for item in suspicious_items:

            word = get_suspicious_word(
                item
            )

            print()
            print(
                f"WORD: {word}"
            )

            if item.get(
                "suggested_split"
            ):

                print(
                    f"SUGGESTED SPLIT: "
                    f"{item['suggested_split']}"
                )

            if item.get(
                "left_frequency"
            ) is not None:

                print(
                    f"LEFT FREQUENCY: "
                    f"{item['left_frequency']}"
                )

            if item.get(
                "right_frequency"
            ) is not None:

                print(
                    f"RIGHT FREQUENCY: "
                    f"{item['right_frequency']}"
                )

            if item.get(
                "evidence_score"
            ) is not None:

                print(
                    f"EVIDENCE SCORE: "
                    f"{item['evidence_score']}"
                )

            print(
                "-" * 80
            )

            word_contexts = contexts.get(
                word,
                [],
            )

            if not word_contexts:

                print(
                    "No context found."
                )

            else:

                for i, context in enumerate(
                    word_contexts,
                    start=1,
                ):

                    print(
                        f"[Context {i}]"
                    )

                    print(
                        context
                    )

                    print()

    # =========================================================================
    # SAVE CLEANED TEXT
    # =========================================================================

    OUTPUT_TEXT_PATH.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    # =========================================================================
    # SAVE JSON REPORT
    # =========================================================================

    json_report = report_to_dict(
        report
    )

    json_report[
        "original_characters"
    ] = original_characters

    json_report[
        "cleaned_characters"
    ] = cleaned_characters

    json_report[
        "characters_removed"
    ] = (
        original_characters
        - cleaned_characters
    )

    json_report["suspicious_joined_words_count"] = len(suspicious_items)
    json_report["suspicious_words"] = suspicious_items  

    OUTPUT_REPORT_PATH.write_text(
        json.dumps(
            json_report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # SAVE SUSPICIOUS REPORT
    # =========================================================================

    save_suspicious_report(
        OUTPUT_SUSPICIOUS_PATH,
        suspicious_items,
        contexts,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"Cleaned text:"
    )

    print(
        OUTPUT_TEXT_PATH
    )

    print()

    print(
        f"Cleaning report:"
    )

    print(
        OUTPUT_REPORT_PATH
    )

    print()

    print(
        f"Suspicious-word report:"
    )

    print(
        OUTPUT_SUSPICIOUS_PATH
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"PDF pages             : {page_count}"
    )

    print(
        f"Original characters   : "
        f"{original_characters:,}"
    )

    print(
        f"Cleaned characters    : "
        f"{cleaned_characters:,}"
    )

    print(
        f"Characters removed    : "
        f"{original_characters - cleaned_characters:,}"
    )

    print(
        f"Suspicious candidates : "
        f"{len(suspicious_items)}"
    )

    print()

    if suspicious_items:

        print(
            "Manual review required "
            "for suspicious candidates."
        )

    else:

        print(
            "No suspicious joined-word "
            "candidates detected."
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()