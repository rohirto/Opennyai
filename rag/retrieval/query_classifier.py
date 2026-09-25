# rag/retrieval/query_classifier.py

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryAnalysis:
    query_type: str = "general"

    preferred_roles: list[str] = field(
        default_factory=list
    )

    excluded_roles: list[str] = field(
        default_factory=list
    )

    keywords: list[str] = field(
        default_factory=list
    )

    court: Optional[str] = None

    case_name: Optional[str] = None

    citation: Optional[str] = None

    # Explicit case/document identity.
    case_query: Optional[str] = None

    # Used by retrieval/reranking.
    entities: tuple[str, ...] = ()

    # Pipeline attaches Qdrant Filter here.
    _qdrant_filter: object = None


class LegalQueryClassifier:

    STOPWORDS = {
        "what",
        "was",
        "is",
        "the",
        "in",
        "of",
        "did",
        "do",
        "does",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "and",
        "or",
        "to",
        "for",
        "under",
        "about",
        "case",
        "judgment",
        "judgement",
        "decision",
        "court",
        "please",
        "tell",
        "me",
    }

    JUDGMENT_TERMS = (
        "judgment",
        "judgement",
        "decision",
        "held",
        "holding",
        "decide",
        "decided",
        "outcome",
        "disposed",
        "dismissed",
        "allowed",
    )

    def classify(
        self,
        query: str,
    ) -> QueryAnalysis:

        query = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        if not query:
            return QueryAnalysis()

        lowered = query.casefold()

        query_type = self._detect_query_type(
            lowered
        )

        case_query = self._extract_case_query(
            query
        )
        
        case_name = _extract_case_name(query)

        keywords = self._keywords(
            lowered
        )

        if case_query:
            entities = (
                case_query,
            )
        else:
            entities = tuple()

        preferred_roles = self._preferred_roles(
            query_type
        )

        return QueryAnalysis(
            query_type=query_type,
            preferred_roles=preferred_roles,
            excluded_roles=[],
            keywords=keywords,
            court=self._extract_court(lowered),
            case_name=case_name,
            citation=self._extract_citation(query),
            case_query=case_query,
            entities=entities,
        )

    # ------------------------------------------------------------------
    # QUERY TYPE
    # ------------------------------------------------------------------

    def _detect_query_type(
        self,
        query: str,
    ) -> str:

        if any(
            phrase in query
            for phrase in (
                "what is the judgment",
                "what is the judgement",
                "judgment in",
                "judgement in",
                "decision in",
                "what did the court decide",
                "what was the decision",
                "what did the court hold",
                "what was held",
                "outcome of",
                "holding in",
            )
        ):
            return "case_lookup"

        if any(
            phrase in query
            for phrase in (
                "why did the court",
                "reasoning",
                "rationale",
                "reason for",
            )
        ):
            return "reasoning"

        if any(
            phrase in query
            for phrase in (
                "procedural history",
                "history of the case",
                "history of proceedings",
            )
        ):
            return "procedural_history"

        if any(
            phrase in query
            for phrase in (
                "interpreted",
                "interpretation",
                "meaning of",
                "construed",
            )
        ):
            return "interpretation"

        if any(
            phrase in query
            for phrase in (
                "application of",
                "applied to",
                "applied in",
            )
        ):
            return "application"

        if any(
            phrase in query
            for phrase in (
                "facts of",
                "what happened in",
                "factual background",
            )
        ):
            return "facts"

        return "general"

    # ------------------------------------------------------------------
    # CASE NAME / DOCUMENT IDENTITY
    # ------------------------------------------------------------------

    def _extract_case_query(
        self,
        query: str,
    ) -> Optional[str]:

        # --------------------------------------------------------------
        # Pattern:
        #
        #   "judgment in Sunil Kumar Case"
        #   "judgement in Sunil Kumar case"
        #   "decision in Sunil Kumar"
        # --------------------------------------------------------------

        patterns = (
            r"(?:judgment|judgement|decision|holding|outcome)"
            r"\s+(?:in|of)\s+"
            r"(.+?)"
            r"(?:\s+case)?$",

            r"(?:what did|what was)\s+"
            r"(?:the\s+)?(?:court|supreme court)"
            r"\s+(?:decide|hold)\s+(?:in|about)\s+"
            r"(.+?)"
            r"(?:\s+case)?$",
        )

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if match:
                value = match.group(1).strip()

                value = re.sub(
                    r"\s+case$",
                    "",
                    value,
                    flags=re.IGNORECASE,
                ).strip()

                if len(value) >= 3:
                    return value

        # --------------------------------------------------------------
        # Conservative fallback for:
        #
        #   "What is the judgment in Sunil Kumar Case?"
        #
        # Remove generic legal words and retain the candidate name.
        # --------------------------------------------------------------

        if (
            "judgment" in query.casefold()
            or "judgement" in query.casefold()
            or "decision" in query.casefold()
        ):

            cleaned = re.sub(
                r"\b(?:what|is|was|the|judgment|judgement|"
                r"decision|in|of|case|about|did|court)\b",
                " ",
                query,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"\s+",
                " ",
                cleaned,
            ).strip(" ?.,:-")

            if len(cleaned) >= 3:
                return cleaned

        return None

    # ------------------------------------------------------------------
    # KEYWORDS
    # ------------------------------------------------------------------

    def _keywords(
        self,
        query: str,
    ) -> list[str]:

        tokens = re.findall(
            r"[A-Za-z0-9]+",
            query,
        )

        return [
            token
            for token in tokens
            if (
                token not in self.STOPWORDS
                and len(token) >= 3
            )
        ]

    # ------------------------------------------------------------------
    # ROLES
    # ------------------------------------------------------------------

    def _preferred_roles(
        self,
        query_type: str,
    ) -> list[str]:

        if query_type == "case_lookup":
            return [
                "RPC",
                "RATIO",
                "ANALYSIS",
                "ARG_RESPONDENT",
                "ARG_PETITIONER",
            ]

        if query_type == "reasoning":
            return [
                "RATIO",
                "ANALYSIS",
            ]

        if query_type == "procedural_history":
            return [
                "PREAMBLE",
                "FACTS",
                "ANALYSIS",
                "RPC",
            ]

        if query_type == "interpretation":
            return [
                "RATIO",
                "ANALYSIS",
            ]

        if query_type == "application":
            return [
                "ANALYSIS",
                "RATIO",
            ]

        return [
            "ANALYSIS",
            "RATIO",
        ]

    # ------------------------------------------------------------------
    # COURT
    # ------------------------------------------------------------------

    def _extract_court(
        self,
        query: str,
    ) -> Optional[str]:

        if "supreme court" in query:
            return "Supreme Court of India"

        if "high court" in query:
            return "High Court"

        return None

    # ------------------------------------------------------------------
    # CITATION
    # ------------------------------------------------------------------

    def _extract_citation(
        self,
        query: str,
    ) -> Optional[str]:

        match = re.search(
            r"\b\d{4}\s+\(\d+\)\s+\w+\s+\d+\b",
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(0)

        return None

def _extract_case_name(query: str) -> str | None:
    """
    Extract a human-readable case name from common legal query forms.

    Examples
    --------
    "What is the judgment in Sunil Kumar B?"
        -> "Sunil Kumar B"

    "What is the judgement in case of Sunil Kumar B?"
        -> "Sunil Kumar B"

    "What did the court decide in Sunil Kumar B case?"
        -> "Sunil Kumar B"
    """

    if not query:
        return None

    text = query.strip()

    # Remove trailing punctuation.
    text = re.sub(
        r"[?!.:,;]+$",
        "",
        text,
    ).strip()

    # Common prefixes.
    prefix_patterns = [
        r"^what\s+is\s+the\s+(?:court\s+)?judg(?:e)?ment\s+in\s+",
        r"^what\s+is\s+the\s+(?:court\s+)?judgment\s+in\s+",
        r"^what\s+is\s+the\s+(?:court\s+)?judgement\s+in\s+",
        r"^what\s+is\s+the\s+judgment\s+in\s+",
        r"^what\s+is\s+the\s+judgement\s+in\s+",
        r"^what\s+is\s+the\s+case\s+of\s+",
        r"^what\s+is\s+the\s+judgment\s+of\s+",
        r"^what\s+is\s+the\s+judgement\s+of\s+",
        r"^tell\s+me\s+about\s+the\s+judgment\s+in\s+",
        r"^tell\s+me\s+about\s+the\s+judgement\s+in\s+",
        r"^what\s+did\s+the\s+court\s+decide\s+in\s+",
        r"^what\s+did\s+the\s+supreme\s+court\s+decide\s+in\s+",
        r"^judgment\s+in\s+",
        r"^judgement\s+in\s+",
    ]

    for pattern in prefix_patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    # Remove "case of" if it remains.
    text = re.sub(
        r"^case\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Remove trailing query descriptors.
    text = re.sub(
        r"\s+(?:court\s+)?judg(?:e)?ment$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"\s+case$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"\s+judgment$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"\s+judgement$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Remove accidental trailing punctuation again.
    text = re.sub(
        r"[?!.:,;]+$",
        "",
        text,
    ).strip()

    # Reject obviously useless results.
    if not text:
        return None

    useless = {
        "the court",
        "the case",
        "this case",
        "the judgment",
        "the judgement",
    }

    if text.casefold() in useless:
        return None

    return text