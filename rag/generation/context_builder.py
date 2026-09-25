"""
Legal RAG Context Builder
=========================

Builds LLM-facing context from reranked legal retrieval results.

Design principles
-----------------
1. RPC is the strongest evidence for the present court's final decision.
2. RLC is lower-court evidence and must not be confused with the
   present court's decision.
3. Decision and case-lookup questions use the same decision-oriented
   evidence assembly.
4. Formal disposition comes first.
5. Operative directions come next.
6. Substantive holdings come next.
7. Supporting judicial reasoning comes last.
8. Party arguments and lower-court outcomes are not allowed to dominate.
9. Context is deliberately bounded to keep small local LLMs responsive.
10. Retrieval metadata is preserved for citations/debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rag.models import RetrievalResult, QueryAnalysis


# =============================================================================
# CONFIGURATION
# =============================================================================

# Keep this bounded because Qwen 2.5 3B on a GTX 1650 is relatively slow.
DEFAULT_MAX_CONTEXT_CHARS = 9000

# Case lookup normally needs the final order + a small amount of reasoning.
CASE_LOOKUP_MAX_CONTEXT_CHARS = 7000

MAX_FORMAL_DISPOSITION = 3
MAX_OPERATIVE_DIRECTIONS = 4
MAX_SUBSTANTIVE_HOLDINGS = 4
MAX_DIRECT_ANALYSIS = 3
MAX_SUPPORTING_EVIDENCE = 2


LOW_PRIORITY_DECISION_ROLES = {
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "FAC",
    "PREAMBLE",
}

LOWER_COURT_ROLES = {
    "RLC",
}


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class AssembledContext:
    text: str

    included_chunk_ids: Tuple[str, ...] = field(default_factory=tuple)
    skipped_chunk_ids: Tuple[str, ...] = field(default_factory=tuple)

    character_count: int = 0

    sections: Dict[str, int] = field(default_factory=dict)

    debug: Dict[str, object] = field(default_factory=dict)


# =============================================================================
# NORMALIZATION
# =============================================================================

def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _lower(text: Optional[str]) -> str:
    return _normalize(text).lower()


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

def _split_sentences(text: str) -> List[str]:
    text = _normalize(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# =============================================================================
# LEGAL SIGNALS
# =============================================================================

CURRENT_COURT_OUTCOME_PATTERNS = [
    r"\bwe\s+allow\b",
    r"\bwe\s+allowed\b",
    r"\bwe\s+accordingly\s+allow\b",
    r"\bwe\s+accordingly\s+dispose\b",
    r"\bwe\s+accordingly\s+disposed\b",
    r"\bwe\s+dismiss\b",
    r"\bwe\s+dismissed\b",
    r"\bwe\s+hold\b",
    r"\bwe\s+have\s+held\b",
    r"\bwe\s+direct\b",
    r"\bwe\s+directed\b",
    r"\bwe\s+order\b",
    r"\bwe\s+ordered\b",
    r"\bwe\s+dispose\b",
    r"\bwe\s+disposed\b",

    r"\bthe\s+appeal[s]?\s+(?:is|are)\s+allowed\b",
    r"\bthe\s+appeal[s]?\s+(?:is|are)\s+dismissed\b",
    r"\bthe\s+petition[s]?\s+(?:is|are)\s+allowed\b",
    r"\bthe\s+petition[s]?\s+(?:is|are)\s+dismissed\b",

    r"\bthe\s+matter\s+is\s+disposed\b",
    r"\bthe\s+matter[s]?\s+are\s+disposed\b",

    r"\bstands\s+dismissed\b",
    r"\bstands\s+allowed\b",

    r"\bis\s+hereby\s+dismissed\b",
    r"\bis\s+hereby\s+allowed\b",
    r"\bare\s+hereby\s+dismissed\b",
    r"\bare\s+hereby\s+allowed\b",

    r"\baccordingly,\s+the\s+appeal\b",
    r"\baccordingly,\s+the\s+petition\b",
]


OPERATIVE_DIRECTION_PATTERNS = [
    r"\bwe\s+direct\b",
    r"\bwe\s+hereby\s+direct\b",
    r"\bwe\s+order\b",
    r"\bwe\s+hereby\s+order\b",

    r"\bshall\s+be\s+entitled\b",
    r"\bshall\s+be\s+given\b",
    r"\bshall\s+have\s+to\b",
    r"\bshall\s+also\s+have\s+to\b",
    r"\bshall\s+be\s+permitted\b",
    r"\bshall\s+be\s+treated\b",
    r"\bshall\s+not\b",

    r"\bis\s+directed\b",
    r"\bare\s+directed\b",

    r"\bwe\s+permit\b",
    r"\bwe\s+grant\b",

    r"\boption[s]?\s+shall\b",
    r"\boption[s]?\s+can\s+be\s+exercised\b",
    r"\bexercise\s+option\b",

    r"\bfurther\s+chance\b",
    r"\bwithin\s+\d+\s+(?:days|months|weeks)\b",
    r"\bfrom\s+today\b",
    r"\bfrom\s+the\s+date\s+of\s+this\s+order\b",
    r"\bwith\s+effect\s+from\b",
    r"\bsubject\s+to\b",
]


HOLDING_PATTERNS = [
    r"\bin\s+our\s+view\b",
    r"\bin\s+our\s+opinion\b",
    r"\bwe\s+are\s+of\s+the\s+view\b",
    r"\bwe\s+are\s+of\s+the\s+opinion\b",
    r"\bwe\s+hold\s+that\b",
    r"\bwe\s+hold\b",
    r"\bwe\s+have\s+already\s+held\b",

    r"\bis\s+valid\b",
    r"\bare\s+valid\b",
    r"\bis\s+invalid\b",
    r"\bare\s+invalid\b",

    r"\bis\s+constitutional\b",
    r"\bis\s+unconstitutional\b",

    r"\bis\s+illegal\b",
    r"\bare\s+illegal\b",

    r"\bis\s+ultra\s+vires\b",
    r"\bare\s+ultra\s+vires\b",

    r"\bdoes\s+not\s+survive\b",
    r"\bwe\s+do\s+not\s+find\s+merit\b",
    r"\bwe\s+find\s+merit\b",
]


OTHER_COURT_PATTERNS = [
    r"\bhigh\s+court\b",
    r"\bdivision\s+bench\b",
    r"\bsingle\s+judge\b",
    r"\blearned\s+judge\b",
    r"\btrial\s+court\b",
]


LOWER_COURT_OUTCOME_PATTERNS = [
    r"\bhigh\s+court\b.*\bset\s+aside\b",
    r"\bhigh\s+court\b.*\bquashed\b",
    r"\bhigh\s+court\b.*\ballowed\b",
    r"\bhigh\s+court\b.*\bdismissed\b",
    r"\bhigh\s+court\b.*\bupheld\b",
    r"\btrial\s+court\b.*\ballowed\b",
    r"\btrial\s+court\b.*\bdismissed\b",
    r"\bjudgment\s+of\s+the\s+high\s+court\b",
    r"\border\s+of\s+the\s+high\s+court\b",
]


# =============================================================================
# SIGNAL HELPERS
# =============================================================================

def _contains_pattern(
    text: str,
    patterns: Sequence[str],
) -> bool:

    value = _lower(text)

    return any(
        re.search(pattern, value)
        for pattern in patterns
    )


def _pattern_count(
    text: str,
    patterns: Sequence[str],
) -> int:

    value = _lower(text)

    return sum(
        1
        for pattern in patterns
        if re.search(pattern, value)
    )


def _current_court_outcome_signal(
    result: RetrievalResult,
) -> float:

    text = result.text or ""
    role = (result.primary_role or "").upper()

    signal = 0.0

    # RPC = present-court ruling.
    if role == "RPC":
        signal += 8.0

    signal += min(
        _pattern_count(
            text,
            CURRENT_COURT_OUTCOME_PATTERNS,
        ) * 2.0,
        8.0,
    )

    signal += min(
        _pattern_count(
            text,
            OPERATIVE_DIRECTION_PATTERNS,
        ) * 0.75,
        3.0,
    )

    signal += min(
        _pattern_count(
            text,
            HOLDING_PATTERNS,
        ) * 0.5,
        2.0,
    )

    signal -= min(
        _pattern_count(
            text,
            LOWER_COURT_OUTCOME_PATTERNS,
        ) * 3.0,
        6.0,
    )

    return max(signal, 0.0)


def _other_court_outcome_signal(
    result: RetrievalResult,
) -> float:

    text = result.text or ""
    role = (result.primary_role or "").upper()

    signal = 0.0

    if role == "RLC":
        signal += 7.0

    signal += min(
        _pattern_count(
            text,
            OTHER_COURT_PATTERNS,
        ) * 1.5,
        6.0,
    )

    signal += min(
        _pattern_count(
            text,
            LOWER_COURT_OUTCOME_PATTERNS,
        ) * 2.0,
        6.0,
    )

    return signal


# =============================================================================
# SENTENCE EXTRACTION
# =============================================================================

def _extract_sentences_matching(
    text: str,
    patterns: Sequence[str],
    max_sentences: int = 5,
) -> List[str]:

    selected = []

    for sentence in _split_sentences(text):

        if _contains_pattern(
            sentence,
            patterns,
        ):
            selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    return selected


def _extract_operational_sentences(
    result: RetrievalResult,
    max_sentences: int = 5,
) -> List[str]:

    return _extract_sentences_matching(
        result.text or "",
        OPERATIVE_DIRECTION_PATTERNS
        + CURRENT_COURT_OUTCOME_PATTERNS,
        max_sentences,
    )


def _extract_holding_sentences(
    result: RetrievalResult,
    max_sentences: int = 5,
) -> List[str]:

    return _extract_sentences_matching(
        result.text or "",
        HOLDING_PATTERNS,
        max_sentences,
    )


# =============================================================================
# EVIDENCE CLASSIFICATION
# =============================================================================

def _classify_evidence(
    result: RetrievalResult,
    query_analysis: QueryAnalysis,
) -> str:

    role = (result.primary_role or "").upper()

    query_type = (
        query_analysis.query_type
        or "general"
    ).lower()

    # -------------------------------------------------------------------------
    # Decision + case lookup
    # -------------------------------------------------------------------------

    if query_type in {"decision", "case_lookup"}:

        # RPC is present-court decision evidence.
        if role == "RPC":

            if (
                _current_court_outcome_signal(result)
                >= 5.0
            ):
                return "formal disposition"

            return "operative directions"

        # RLC is always lower-court material.
        if role == "RLC":
            return "lower court outcome"

        current_signal = (
            _current_court_outcome_signal(result)
        )

        other_signal = (
            _other_court_outcome_signal(result)
        )

        # Protect against reproduced lower-court orders.
        if (
            other_signal > current_signal
            and other_signal >= 3.0
        ):
            return "lower court outcome"

        if (
            current_signal >= 7.0
            and current_signal > other_signal
        ):
            return "formal disposition"

        if _contains_pattern(
            result.text or "",
            OPERATIVE_DIRECTION_PATTERNS,
        ):
            if other_signal < 3.0:
                return "operative directions"

        if _contains_pattern(
            result.text or "",
            HOLDING_PATTERNS,
        ):
            if other_signal < 5.0:
                return "substantive holding"

        if role in {
            "ANALYSIS",
            "RATIO",
            "STA",
        }:
            return "direct analysis"

        if role in {
            "ARG_PETITIONER",
            "ARG_RESPONDENT",
        }:
            return "party arguments"

        if role in {
            "FAC",
            "PREAMBLE",
        }:
            return "background"

        if role in {
            "PRE_RELIED",
            "PRE_NOT_RELIED",
        }:
            return "precedent"

        return "supporting evidence"

    # -------------------------------------------------------------------------
    # Facts
    # -------------------------------------------------------------------------

    if query_type == "facts":

        if role in {"FAC", "RLC"}:
            return "case facts"

        if role in {"ANALYSIS", "STA"}:
            return "procedural history"

        if role in {
            "ARG_PETITIONER",
            "ARG_RESPONDENT",
        }:
            return "party arguments"

        return "supporting evidence"

    # -------------------------------------------------------------------------
    # Precedent
    # -------------------------------------------------------------------------

    if query_type == "precedent":

        if role in {
            "PRE_RELIED",
            "PRE_NOT_RELIED",
            "RATIO",
        }:
            return "precedent"

        if role == "ANALYSIS":
            return "direct analysis"

        return "supporting evidence"

    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------

    if query_type == "arguments":

        if role in {
            "ARG_PETITIONER",
            "ARG_RESPONDENT",
        }:
            return "party arguments"

        if role == "ANALYSIS":
            return "judicial response"

        return "supporting evidence"

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------

    if role == "RPC":
        return "formal disposition"

    if role == "RLC":
        return "lower court outcome"

    if role in {
        "ANALYSIS",
        "RATIO",
        "STA",
    }:
        return "direct analysis"

    if role == "FAC":
        return "case facts"

    if role in {
        "ARG_PETITIONER",
        "ARG_RESPONDENT",
    }:
        return "party arguments"

    if role in {
        "PRE_RELIED",
        "PRE_NOT_RELIED",
    }:
        return "precedent"

    return "supporting evidence"


# =============================================================================
# EVIDENCE PRIORITY
# =============================================================================

def _evidence_priority(
    category: str,
    result: RetrievalResult,
    query_analysis: QueryAnalysis,
) -> float:

    base = {
        "formal disposition": 100.0,
        "operative directions": 92.0,
        "substantive holding": 86.0,
        "direct analysis": 72.0,
        "judicial response": 68.0,
        "case facts": 52.0,
        "procedural history": 45.0,
        "party arguments": 28.0,
        "precedent": 25.0,
        "supporting evidence": 20.0,
        "background": 15.0,
        "lower court outcome": 5.0,
    }.get(category, 10.0)

    role = (result.primary_role or "").upper()

    if role == "RPC":
        base += 20.0

    if role == "RATIO":
        base += 10.0

    if role == "ANALYSIS":
        base += 4.0

    base += min(
        _current_court_outcome_signal(result) * 2.0,
        20.0,
    )

    if query_analysis.query_type in {
        "decision",
        "case_lookup",
    }:
        base -= min(
            _other_court_outcome_signal(result) * 5.0,
            35.0,
        )

    if result.vector_score is not None:
        base += (
            max(
                0.0,
                min(
                    float(result.vector_score),
                    1.0,
                ),
            )
            * 10.0
        )

    return base


# =============================================================================
# FORMATTING
# =============================================================================

def _format_result_header(
    result: RetrievalResult,
    evidence_role: str,
) -> str:

    primary_role = (
        result.primary_role
        or "UNKNOWN"
    )

    return (
        f"[Evidence Role: {evidence_role}]\n"
        f"[Chunk: {result.chunk_index}]\n"
        f"[Role: {primary_role}]\n"
        f"[Characters: "
        f"{result.char_start}-{result.char_end}]\n"
    )


def _format_result(
    result: RetrievalResult,
    evidence_role: str,
    text_override: Optional[str] = None,
) -> str:

    body = _normalize(
        text_override
        if text_override is not None
        else result.text
    )

    return (
        _format_result_header(
            result,
            evidence_role,
        )
        + "\n"
        + body
        + "\n"
    )


# =============================================================================
# DEDUPLICATION
# =============================================================================

def _deduplicate_texts(
    texts: Sequence[str],
) -> List[str]:

    seen = set()
    output = []

    for text in texts:

        normalized = re.sub(
            r"\s+",
            " ",
            text.strip().lower(),
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        output.append(text.strip())

    return output


# =============================================================================
# APPEND WITH BUDGET
# =============================================================================

def _append_block(
    sections: List[str],
    included: List[RetrievalResult],
    included_ids: set,
    result: RetrievalResult,
    evidence_role: str,
    max_chars: int,
    text_override: Optional[str] = None,
) -> bool:

    if result.chunk_id in included_ids:
        return False

    block = _format_result(
        result=result,
        evidence_role=evidence_role,
        text_override=text_override,
    )

    current_chars = sum(
        len(section)
        for section in sections
    )

    additional_chars = len(block) + 2

    if (
        current_chars + additional_chars
        > max_chars
    ):
        return False

    sections.append(block)

    included.append(result)
    included_ids.add(result.chunk_id)

    return True


# =============================================================================
# DECISION / CASE LOOKUP CONTEXT
# =============================================================================

def _assemble_decision_context(
    results: Sequence[RetrievalResult],
    query_analysis: QueryAnalysis,
    max_chars: int,
) -> AssembledContext:

    classified = []

    for result in results:

        category = _classify_evidence(
            result,
            query_analysis,
        )

        priority = _evidence_priority(
            category,
            result,
            query_analysis,
        )

        classified.append(
            (
                priority,
                category,
                result,
            )
        )

    classified.sort(
        key=lambda item: (
            -item[0],
            item[2].chunk_index,
        )
    )

    buckets: Dict[
        str,
        List[Tuple[float, RetrievalResult]],
    ] = {}

    for priority, category, result in classified:

        buckets.setdefault(
            category,
            [],
        ).append(
            (priority, result)
        )

    for category in buckets:

        buckets[category].sort(
            key=lambda item: (
                -item[0],
                item[1].chunk_index,
            )
        )

    sections: List[str] = []

    included: List[RetrievalResult] = []

    included_ids: set = set()

    section_counts: Dict[str, int] = {}

    # -------------------------------------------------------------------------
    # 1. FORMAL DISPOSITION
    # -------------------------------------------------------------------------

    formal_candidates = buckets.get(
        "formal disposition",
        [],
    )

    if formal_candidates:

        sections.append(
            "=== COURT DECISION / FORMAL DISPOSITION ===\n"
        )

        added = 0

        for _, result in formal_candidates:

            if added >= MAX_FORMAL_DISPOSITION:
                break

            if _append_block(
                sections,
                included,
                included_ids,
                result,
                "formal disposition",
                max_chars,
            ):
                added += 1

        section_counts[
            "formal_disposition"
        ] = added

    # -------------------------------------------------------------------------
    # 2. OPERATIVE DIRECTIONS
    # -------------------------------------------------------------------------

    operative_candidates = buckets.get(
        "operative directions",
        [],
    )

    if operative_candidates:

        sections.append(
            "=== OPERATIVE DIRECTIONS ===\n"
        )

        added = 0

        for _, result in operative_candidates:

            if added >= MAX_OPERATIVE_DIRECTIONS:
                break

            sentences = (
                _extract_operational_sentences(
                    result,
                    max_sentences=5,
                )
            )

            if sentences:

                extracted = "\n".join(
                    _deduplicate_texts(
                        sentences
                    )
                )

                success = _append_block(
                    sections,
                    included,
                    included_ids,
                    result,
                    "operative directions",
                    max_chars,
                    extracted,
                )

            else:

                success = _append_block(
                    sections,
                    included,
                    included_ids,
                    result,
                    "operative directions",
                    max_chars,
                )

            if success:
                added += 1

        section_counts[
            "operative_directions"
        ] = added

    # -------------------------------------------------------------------------
    # 3. SUBSTANTIVE HOLDINGS
    # -------------------------------------------------------------------------

    holding_candidates = buckets.get(
        "substantive holding",
        [],
    )

    if holding_candidates:

        sections.append(
            "=== SUBSTANTIVE HOLDINGS ===\n"
        )

        added = 0

        for _, result in holding_candidates:

            if added >= MAX_SUBSTANTIVE_HOLDINGS:
                break

            sentences = (
                _extract_holding_sentences(
                    result,
                    max_sentences=5,
                )
            )

            if sentences:

                extracted = "\n".join(
                    _deduplicate_texts(
                        sentences
                    )
                )

                success = _append_block(
                    sections,
                    included,
                    included_ids,
                    result,
                    "substantive holding",
                    max_chars,
                    extracted,
                )

            else:

                success = _append_block(
                    sections,
                    included,
                    included_ids,
                    result,
                    "substantive holding",
                    max_chars,
                )

            if success:
                added += 1

        section_counts[
            "substantive_holdings"
        ] = added

    # -------------------------------------------------------------------------
    # 4. DIRECT ANALYSIS
    # -------------------------------------------------------------------------

    analysis_candidates = buckets.get(
        "direct analysis",
        [],
    )

    if analysis_candidates:

        sections.append(
            "=== SUPPORTING JUDICIAL ANALYSIS ===\n"
        )

        added = 0

        for _, result in analysis_candidates:

            if added >= MAX_DIRECT_ANALYSIS:
                break

            if _append_block(
                sections,
                included,
                included_ids,
                result,
                "direct analysis",
                max_chars,
            ):
                added += 1

        section_counts[
            "direct_analysis"
        ] = added

    # -------------------------------------------------------------------------
    # 5. SUPPORTING EVIDENCE
    # -------------------------------------------------------------------------

    supporting_candidates = buckets.get(
        "supporting evidence",
        [],
    )

    if supporting_candidates:

        sections.append(
            "=== SUPPORTING EVIDENCE ===\n"
        )

        added = 0

        for _, result in supporting_candidates:

            if added >= MAX_SUPPORTING_EVIDENCE:
                break

            if _append_block(
                sections,
                included,
                included_ids,
                result,
                "supporting evidence",
                max_chars,
            ):
                added += 1

        section_counts[
            "supporting_evidence"
        ] = added

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Lower court outcomes are deliberately NOT inserted into the primary
    # decision context.
    # -------------------------------------------------------------------------

    context_text = "\n".join(
        section.strip()
        for section in sections
        if section.strip()
    ).strip()

    skipped_ids = tuple(
        result.chunk_id
        for _, _, result in classified
        if result.chunk_id not in included_ids
    )

    included_ids_tuple = tuple(
        result.chunk_id
        for result in included
    )

    classification_debug = []

    for priority, category, result in classified:

        classification_debug.append(
            {
                "chunk_index": result.chunk_index,
                "chunk_id": result.chunk_id,
                "primary_role": result.primary_role,
                "category": category,
                "priority": round(
                    priority,
                    3,
                ),
                "current_court_outcome_signal":
                    round(
                        _current_court_outcome_signal(
                            result
                        ),
                        3,
                    ),
                "other_court_outcome_signal":
                    round(
                        _other_court_outcome_signal(
                            result
                        ),
                        3,
                    ),
                "vector_score": (
                    round(
                        result.vector_score,
                        6,
                    )
                    if result.vector_score is not None
                    else None
                ),
            }
        )

    return AssembledContext(
        text=context_text,
        included_chunk_ids=(
            included_ids_tuple
        ),
        skipped_chunk_ids=skipped_ids,
        character_count=len(context_text),
        sections=section_counts,
        debug={
            "query_type":
                query_analysis.query_type,
            "classification":
                classification_debug,
            "max_context_chars":
                max_chars,
        },
    )


# =============================================================================
# GENERIC CONTEXT
# =============================================================================

def _assemble_generic_context(
    results: Sequence[RetrievalResult],
    query_analysis: QueryAnalysis,
    max_chars: int,
) -> AssembledContext:

    classified = []

    for result in results:

        category = _classify_evidence(
            result,
            query_analysis,
        )

        priority = _evidence_priority(
            category,
            result,
            query_analysis,
        )

        classified.append(
            (
                priority,
                category,
                result,
            )
        )

    classified.sort(
        key=lambda item: (
            -item[0],
            item[2].chunk_index,
        )
    )

    sections: List[str] = []

    included: List[RetrievalResult] = []

    included_ids: set = set()

    section_counts: Dict[str, int] = {}

    for _, category, result in classified:

        if _append_block(
            sections,
            included,
            included_ids,
            result,
            category,
            max_chars,
        ):
            section_counts[category] = (
                section_counts.get(
                    category,
                    0,
                )
                + 1
            )

    context_text = "\n".join(
        section.strip()
        for section in sections
        if section.strip()
    ).strip()

    skipped_ids = tuple(
        result.chunk_id
        for _, _, result in classified
        if result.chunk_id not in included_ids
    )

    included_ids_tuple = tuple(
        result.chunk_id
        for result in included
    )

    return AssembledContext(
        text=context_text,
        included_chunk_ids=(
            included_ids_tuple
        ),
        skipped_chunk_ids=skipped_ids,
        character_count=len(context_text),
        sections=section_counts,
        debug={
            "query_type":
                query_analysis.query_type,
            "max_context_chars":
                max_chars,
        },
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def assemble_context(
    results: Sequence[RetrievalResult],
    query_analysis: QueryAnalysis,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> AssembledContext:

    if not results:

        return AssembledContext(
            text="",
            character_count=0,
            debug={
                "query_type":
                    query_analysis.query_type,
                "message":
                    "No retrieval results.",
            },
        )

    query_type = (
        query_analysis.query_type
        or "general"
    ).lower()

    # IMPORTANT:
    # A case lookup asking for a judgment is still a decision-oriented
    # question. Do not send it through the generic context path.
    if query_type in {
        "decision",
        "case_lookup",
    }:

        effective_max_chars = max_context_chars

        if query_type == "case_lookup":

            effective_max_chars = min(
                max_context_chars,
                CASE_LOOKUP_MAX_CONTEXT_CHARS,
            )

        return _assemble_decision_context(
            results=results,
            query_analysis=query_analysis,
            max_chars=effective_max_chars,
        )

    return _assemble_generic_context(
        results=results,
        query_analysis=query_analysis,
        max_chars=max_context_chars,
    )


__all__ = [
    "AssembledContext",
    "assemble_context",
]