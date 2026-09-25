# rag/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# QUERY
# ============================================================================

@dataclass
class QueryAnalysis:
    """
    Structured interpretation of a user query.

    This is intentionally heuristic in v1.
    A learned/LLM classifier can replace it later without changing
    the rest of the pipeline.
    """

    original_query: str

    query_type: str = "general"

    # Legal rhetorical roles that should receive higher retrieval weight.
    preferred_roles: List[str] = field(default_factory=list)

    excluded_roles: List[str] = field(default_factory=list)

    # Extracted legal entities / concepts.
    entities: Dict[str, List[str]] = field(default_factory=dict)

    # Simple keyword representation.
    keywords: List[str] = field(default_factory=list)

    # Optional metadata filters.
    court: Optional[str] = None
    case_name: Optional[str] = None
    citation: Optional[str] = None
    judgment_date: Optional[str] = None

    # Whether the query is asking specifically for a final determination.
    asks_for_decision: bool = False

    # Whether the query is asking about a precedent.
    asks_about_precedent: bool = False

    # Whether the query asks about facts/events.
    asks_about_facts: bool = False


# ============================================================================
# RETRIEVAL
# ============================================================================

@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    chunk_index: int

    text: str

    score: float
    primary_role: str
    roles: List[str]

    char_start: int
    char_end: int

    source_file: Optional[str] = None
    case_name: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    judgment_date: Optional[str] = None

    entities: Dict[str, List[str]] = field(
        default_factory=dict
    )

    summary_context: Dict[str, str] = field(
        default_factory=dict
    )

    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None

    retrieval_signals: Dict[str, float] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

# ============================================================================
# GENERATION
# ============================================================================

@dataclass
class SourceCitation:
    """
    Source information exposed to the user.
    """

    source_id: str

    document_id: str
    chunk_index: int

    source_file: Optional[str]

    case_name: Optional[str]
    citation: Optional[str]
    court: Optional[str]
    judgment_date: Optional[str]

    char_start: int
    char_end: int

    role: str


@dataclass
class GeneratedAnswer:
    answer: str
    sources: List[SourceCitation] = field(
        default_factory=list
    )

    query: Optional[str] = None
    model: Optional[str] = None

    retrieval_count: int = 0

    # NEW
    retrieved_results: List[RetrievalResult] = field(
        default_factory=list
    )

    debug: Dict[str, Any] = field(
        default_factory=dict
    )