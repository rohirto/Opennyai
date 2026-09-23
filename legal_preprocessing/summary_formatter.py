from pathlib import Path


# Preferred presentation order.
SECTION_ORDER = [
    "PREAMBLE",
    "FACTS",
    "ISSUE",
    "ARGUMENTS",
    "ANALYSIS",
    "DECISION",
]


def format_summary(summarizer_result):
    """
    Convert OpenNyAI extractive summarizer output into
    a human-readable legal summary.

    The original extracted text is preserved. This function
    only reorganizes and formats it.
    """

    if not summarizer_result:
        return ""

    # Summarizer returns a list, normally containing one document.
    document = summarizer_result[0]

    summaries = document.get("summaries", {})

    lines = []

    lines.append("=" * 80)
    lines.append("LEGAL JUDGMENT SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    for section in SECTION_ORDER:

        # OpenNyAI currently uses "decision" and "arguments"
        # in some outputs rather than uppercase labels.
        section_key = section

        if section == "DECISION":
            section_key = "decision"
        elif section == "ARGUMENTS":
            section_key = "arguments"

        text = summaries.get(section_key)

        if not text:
            continue

        text = str(text).strip()

        if not text:
            continue

        lines.append("-" * 80)
        lines.append(section)
        lines.append("-" * 80)
        lines.append("")
        lines.append(text)
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def save_summary(path: Path, summarizer_result):
    """
    Format and save the summarizer result as summary.txt.
    """

    summary_text = format_summary(summarizer_result)

    with path.open("w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text