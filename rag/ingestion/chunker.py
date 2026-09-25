"""
Legal chunker for the OpenNyAI-based legal RAG pipeline.

Input
-----
LegalDocument

Output
------
List[LegalChunk]

Design
------
RRC
    -> provides rhetorical/legal structure

Legal paragraphs
    -> primary chunking boundary

NER
    -> enriches chunks with entity metadata

Summarizer
    -> provides document-level contextual information

Source text
    -> remains authoritative

Important
---------
This chunker does NOT:
    - generate embeddings
    - communicate with Qdrant
    - call an LLM
    - modify source text
"""

from __future__ import annotations

import json
import re

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from rag.ingestion.document_builder import (
    LegalAnnotation,
    LegalDocument,
    LegalEntity,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MAX_CHARS = 2200

# Retained for API compatibility.
#
# IMPORTANT:
# This is NOT used as a blanket "merge every chunk below 100 chars" rule.
# Legally meaningful short passages are allowed to remain independent.
DEFAULT_MIN_STANDALONE_CHARS = 100


SOFT_ROLES = {
    "NONE",
    "PREAMBLE",
}


HARD_ROLES = {
    "RATIO",
    "RPC",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "ANALYSIS",
    "FAC",
    "STA",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "RLC",
    "ISSUE",
}


# ============================================================================
# Regular expressions
# ============================================================================

# Judgment paragraphs:
#
#   2. In this judgment...
#   13. The judgment...
#   44. We accordingly hold...
#
_MAJOR_PARAGRAPH_RE = re.compile(
    r"^\s*(\d{1,3})\.\s+",
)


# Legal enumeration markers:
#
#   (i)
#   (ii)
#   (iii)
#   (iv)
#   ...
#
_ROMAN_ITEM_RE = re.compile(
    r"^\s*\("
    r"(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)"
    r"\)\s*",
    re.IGNORECASE,
)


# Alphabetic legal lists:
#
#   (a)
#   (b)
#   (c)
#
_ALPHA_ITEM_RE = re.compile(
    r"^\s*\([a-z]\)\s*",
    re.IGNORECASE,
)


# Numeric lists:
#
#   (1)
#   (2)
#
_NUMERIC_ITEM_RE = re.compile(
    r"^\s*\(\d+\)\s*",
)


# Common section / heading fragments.
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"definitions?"
    r"|explanation"
    r"|provided that"
    r"|provided further"
    r"|provided also"
    r"|proviso"
    r")"
    r"\s*:?\s*$",
    re.IGNORECASE,
)


# Text which is effectively only a structural marker.
_STRUCTURAL_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"\(\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\s*\)"
    r"|\(\s*[a-z]\s*\)"
    r"|\(\s*\d+\s*\)"
    r"|\d+\."
    r")\s*$",
    re.IGNORECASE,
)


# ============================================================================
# Data model
# ============================================================================

@dataclass
class LegalChunk:
    """
    One enriched legal retrieval unit.

    text
        Authoritative source text represented by this chunk.

    embedding_text
        Compact derived representation used for embedding.

    source_char_count
        Character span in the authoritative source document.

    text_char_count
        Actual character count of the chunk text.

    entities
        NER entities grouped by label.

    summary_context
        Relevant document-level summary information.
    """

    chunk_index: int

    text: str

    embedding_text: str

    primary_role: str

    roles: List[str]

    role_sequence: List[str]

    annotation_ids: List[str]

    char_start: int

    char_end: int

    source_char_count: int

    text_char_count: int

    annotation_count: int

    entities: Dict[str, List[str]]

    entity_ids: List[str]

    summary_context: Dict[str, str]

    document_id: str

    case_name: str | None

    citation: str | None

    court: str | None

    judgment_date: str | None

    page_start: int | None = None

    page_end: int | None = None


# ============================================================================
# Basic helpers
# ============================================================================

def _annotation_length(
    annotation: LegalAnnotation,
) -> int:
    """Return the authoritative source-span length."""

    return max(
        0,
        annotation.char_end - annotation.char_start,
    )


def _source_text_for_annotations(
    document: LegalDocument,
    annotations: List[LegalAnnotation],
) -> str:
    """
    Extract authoritative source text.

    NEVER reconstruct source text from annotation.text.
    """

    if not annotations:
        return ""

    char_start = annotations[0].char_start
    char_end = annotations[-1].char_end

    return document.text[char_start:char_end]


def _paragraph_number(
    text: str,
) -> int | None:
    """
    Detect a numbered judgment paragraph.

    Examples:

        2. In this judgment...
        13. The judgment...
        44. We accordingly hold...

    Does not treat (i), (ii), etc. as paragraph numbers.
    """

    match = _MAJOR_PARAGRAPH_RE.match(
        text.strip()
    )

    if not match:
        return None

    return int(match.group(1))


def _is_list_marker(
    text: str,
) -> bool:
    """Return True for legal enumeration markers."""

    text = text.strip()

    return bool(
        _ROMAN_ITEM_RE.match(text)
        or _ALPHA_ITEM_RE.match(text)
        or _NUMERIC_ITEM_RE.match(text)
    )


def _is_structural_only(
    text: str,
) -> bool:
    """
    Detect annotations which contain only a structural marker.

    Examples:

        (1)
        (i)
        17.

    These should generally not become independent retrieval chunks.
    """

    return bool(
        _STRUCTURAL_ONLY_RE.match(
            text.strip()
        )
    )


def _is_heading_only(
    text: str,
) -> bool:
    """Detect standalone legal headings."""

    return bool(
        _HEADING_RE.match(
            text.strip()
        )
    )


def _starts_list_item(
    text: str,
) -> bool:
    """
    Detect a list item which contains actual content.

    Example:

        (i) The Employee's Pension...
    """

    stripped = text.strip()

    return bool(
        _ROMAN_ITEM_RE.match(stripped)
        or _ALPHA_ITEM_RE.match(stripped)
        or _NUMERIC_ITEM_RE.match(stripped)
    )


def _is_same_list_family(
    first_text: str,
    second_text: str,
) -> bool:
    """
    Determine whether two annotations belong to the same enumeration family.
    """

    first = first_text.strip()
    second = second_text.strip()

    families = []

    for text in (first, second):

        if _ROMAN_ITEM_RE.match(text):
            families.append("roman")

        elif _ALPHA_ITEM_RE.match(text):
            families.append("alpha")

        elif _NUMERIC_ITEM_RE.match(text):
            families.append("numeric")

        else:
            families.append(None)

    return (
        families[0] is not None
        and families[0] == families[1]
    )


# ============================================================================
# Entity handling
# ============================================================================

def _entity_overlaps_chunk(
    entity: LegalEntity,
    chunk_start: int,
    chunk_end: int,
) -> bool:
    """Return True when an entity overlaps the chunk span."""

    if entity.char_start is None:
        return False

    entity_start = entity.char_start

    entity_end = (
        entity.char_end
        if entity.char_end is not None
        else entity.char_start
    )

    return (
        entity_start < chunk_end
        and entity_end > chunk_start
    )


def _get_chunk_entities(
    document: LegalDocument,
    chunk_start: int,
    chunk_end: int,
) -> tuple[Dict[str, List[str]], List[str]]:
    """
    Return NER entities overlapping the chunk.
    """

    entities: Dict[str, List[str]] = {}

    entity_ids: List[str] = []

    for entity in document.entities:

        if not _entity_overlaps_chunk(
            entity,
            chunk_start,
            chunk_end,
        ):
            continue

        label = entity.label

        value = (
            entity.normalized_name
            or entity.text
        )

        value = value.strip()

        if not value:
            continue

        entities.setdefault(
            label,
            [],
        )

        if value not in entities[label]:
            entities[label].append(value)

        entity_ids.append(
            entity.entity_id
        )

    return entities, entity_ids


# ============================================================================
# Summary handling
# ============================================================================

def _summary_context_for_role(
    document: LegalDocument,
    role: str,
) -> Dict[str, str]:
    """
    Select compact document-level summary context.

    The full summary is NOT inserted into the embedding text.
    """

    summary = document.summary

    context: Dict[str, str] = {}

    if role in {
        "PREAMBLE",
        "FAC",
        "RLC",
    }:

        if summary.preamble:
            context["preamble"] = summary.preamble

        if summary.facts:
            context["facts"] = summary.facts

    elif role in {
        "ARG_PETITIONER",
        "ARG_RESPONDENT",
    }:

        if summary.arguments:
            context["arguments"] = summary.arguments

    elif role in {
        "ANALYSIS",
        "RATIO",
        "RPC",
    }:

        if summary.analysis:
            context["analysis"] = summary.analysis

        if summary.decision:
            context["decision"] = summary.decision

    elif role == "PRE_RELIED":

        if summary.analysis:
            context["analysis"] = summary.analysis

    return context


# ============================================================================
# Embedding representation
# ============================================================================

def _build_embedding_text(
    document: LegalDocument,
    chunk_text: str,
    role: str,
    entities: Dict[str, List[str]],
) -> str:
    """
    Build compact embedding representation.

    Important:
        The authoritative chunk text remains the dominant content.

    We do NOT inject the entire document summary.
    """

    parts: List[str] = []

    if document.case_name:
        parts.append(
            f"Case: {document.case_name}"
        )

    if document.citation:
        parts.append(
            f"Citation: {document.citation}"
        )

    if document.court:
        parts.append(
            f"Court: {document.court}"
        )

    parts.append(
        f"Legal role: {role}"
    )

    if entities:

        entity_parts = []

        for label, values in entities.items():

            if not values:
                continue

            entity_parts.append(
                f"{label}: "
                + ", ".join(values)
            )

        if entity_parts:

            parts.append(
                "Entities: "
                + " | ".join(entity_parts)
            )

    parts.append(
        "Text:\n"
        + chunk_text.strip()
    )

    return "\n".join(parts)


# ============================================================================
# Chunk construction
# ============================================================================

def _create_chunk(
    document: LegalDocument,
    annotations: List[LegalAnnotation],
    chunk_index: int,
) -> LegalChunk:
    """
    Construct one LegalChunk.

    The text is always reconstructed from the authoritative document.
    """

    if not annotations:
        raise ValueError(
            "Cannot create chunk from empty annotations."
        )

    annotations = sorted(
        annotations,
        key=lambda annotation: annotation.char_start,
    )

    char_start = annotations[0].char_start
    char_end = annotations[-1].char_end

    text = document.text[
        char_start:char_end
    ]

    roles_sequence = [
        annotation.role
        for annotation in annotations
    ]

    roles = list(
        dict.fromkeys(
            roles_sequence
        )
    )

    primary_role = roles[0]

    entities, entity_ids = _get_chunk_entities(
        document,
        char_start,
        char_end,
    )

    summary_context = _summary_context_for_role(
        document,
        primary_role,
    )

    embedding_text = _build_embedding_text(
        document=document,
        chunk_text=text,
        role=primary_role,
        entities=entities,
    )

    return LegalChunk(
        chunk_index=chunk_index,

        text=text,

        embedding_text=embedding_text,

        primary_role=primary_role,

        roles=roles,

        role_sequence=roles_sequence,

        annotation_ids=[
            annotation.annotation_id
            for annotation in annotations
        ],

        char_start=char_start,

        char_end=char_end,

        source_char_count=(
            char_end - char_start
        ),

        text_char_count=len(text),

        annotation_count=len(
            annotations
        ),

        entities=entities,

        entity_ids=entity_ids,

        summary_context=summary_context,

        document_id=document.document_id,

        case_name=document.case_name,

        citation=document.citation,

        court=document.court,

        judgment_date=document.judgment_date,
    )


# ============================================================================
# Annotation lookup
# ============================================================================

def _annotation_lookup(
    document: LegalDocument,
) -> Dict[str, LegalAnnotation]:

    return {
        annotation.annotation_id: annotation
        for annotation in document.annotations
    }


# ============================================================================
# Block construction
# ============================================================================

def _build_paragraph_blocks(
    document: LegalDocument,
) -> List[tuple[int | None, List[LegalAnnotation]]]:
    """
    Group RRC annotations into numbered legal paragraphs.

    RRC role transitions alone do NOT create chunks.

    This is important because a legal paragraph may legitimately contain:

        ANALYSIS
        PRE_RELIED
        ANALYSIS
        RPC

    while still representing one coherent legal proposition.
    """

    annotations = document.annotations

    if not annotations:
        return []

    blocks: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    current: List[LegalAnnotation] = []

    current_paragraph: int | None = None

    for annotation in annotations:

        text = annotation.text.strip()

        if not text:
            continue

        detected = _paragraph_number(
            text
        )

        if (
            detected is not None
            and current
        ):

            blocks.append(
                (
                    current_paragraph,
                    current,
                )
            )

            current = []

        if detected is not None:
            current_paragraph = detected

        current.append(
            annotation
        )

    if current:

        blocks.append(
            (
                current_paragraph,
                current,
            )
        )

    return blocks


# ============================================================================
# Enumeration refinement
# ============================================================================

def _split_enumerated_block(
    paragraph_number: int | None,
    annotations: List[LegalAnnotation],
) -> List[
    tuple[int | None, List[LegalAnnotation]]
]:
    """
    Refine a paragraph containing an explicit legal enumeration.

    Unlike the previous implementation, this is NOT hardcoded to paragraph 44.

    Examples handled:

        (i) ...
        (ii) ...
        (iii) ...
        (iv) ...

    or:

        (a) ...
        (b) ...
        (c) ...

    Each item becomes a logical retrieval unit.

    Important:
        A marker-only annotation such as "(iii)" is attached to the
        following content rather than becoming a standalone chunk.
    """

    if not annotations:
        return []

    has_list = any(
        _starts_list_item(
            annotation.text
        )
        for annotation in annotations
    )

    if not has_list:
        return [
            (
                paragraph_number,
                annotations,
            )
        ]

    result: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    current: List[LegalAnnotation] = []

    current_family: str | None = None

    for annotation in annotations:

        text = annotation.text.strip()

        is_marker = _starts_list_item(
            text
        )

        # ---------------------------------------------------------------
        # Determine list family
        # ---------------------------------------------------------------

        family = None

        if _ROMAN_ITEM_RE.match(text):
            family = "roman"

        elif _ALPHA_ITEM_RE.match(text):
            family = "alpha"

        elif _NUMERIC_ITEM_RE.match(text):
            family = "numeric"

        # ---------------------------------------------------------------
        # New list item
        # ---------------------------------------------------------------

        if (
            is_marker
            and current
            and family == current_family
        ):

            # If current consists only of a previous structural marker,
            # do NOT create a standalone chunk.
            if not all(
                _is_structural_only(
                    item.text
                )
                for item in current
            ):

                result.append(
                    (
                        paragraph_number,
                        current,
                    )
                )

            current = []

        # ---------------------------------------------------------------
        # Marker-only item
        #
        # Keep it with subsequent text.
        # ---------------------------------------------------------------

        if (
            is_marker
            and _is_structural_only(text)
        ):

            current.append(
                annotation
            )

            current_family = family

            continue

        # ---------------------------------------------------------------
        # Actual list item
        # ---------------------------------------------------------------

        if is_marker:

            current_family = family

        current.append(
            annotation
        )

    if current:

        result.append(
            (
                paragraph_number,
                current,
            )
        )

    return result


# ============================================================================
# Structural fragment repair
# ============================================================================

def _repair_structural_fragments(
    document: LegalDocument,
    blocks: List[
        tuple[int | None, List[LegalAnnotation]]
    ],
) -> List[
    tuple[int | None, List[LegalAnnotation]]
]:
    """
    Repair structural fragments such as:

        (1)
        (iii)
        17.
        Definitions.

    without blindly merging all short chunks.

    Strategy:

        marker/heading-only block
            -> attach to following block

    Otherwise:

        keep legally meaningful short material intact.
    """

    if not blocks:
        return []

    repaired: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    i = 0

    while i < len(blocks):

        paragraph_number, block = blocks[i]

        if not block:
            i += 1
            continue

        text = " ".join(
            annotation.text.strip()
            for annotation in block
        ).strip()

        structural = (
            _is_structural_only(text)
            or _is_heading_only(text)
        )

        if (
            structural
            and i + 1 < len(blocks)
        ):

            next_paragraph, next_block = (
                blocks[i + 1]
            )

            # Attach only when the next block is adjacent in document
            # structure. This prevents unrelated sections from being merged.
            combined = [
                *block,
                *next_block,
            ]

            repaired.append(
                (
                    next_paragraph
                    if next_paragraph is not None
                    else paragraph_number,
                    combined,
                )
            )

            i += 2
            continue

        repaired.append(
            (
                paragraph_number,
                block,
            )
        )

        i += 1

    return repaired


# ============================================================================
# Oversized block splitting
# ============================================================================

def _split_oversized_block(
    document: LegalDocument,
    paragraph_number: int | None,
    annotations: List[LegalAnnotation],
    max_chars: int,
) -> List[
    tuple[int | None, List[LegalAnnotation]]
]:
    """
    Split an oversized legal block.

    We split ONLY at RRC annotation boundaries.

    This means we never arbitrarily cut source text in the middle of
    an annotation.
    """

    if not annotations:
        return []

    result: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    current: List[LegalAnnotation] = []

    for annotation in annotations:

        if not current:

            current = [
                annotation
            ]

            continue

        current_start = (
            current[0].char_start
        )

        proposed_end = (
            annotation.char_end
        )

        proposed_length = (
            proposed_end
            - current_start
        )

        if (
            proposed_length > max_chars
        ):

            result.append(
                (
                    paragraph_number,
                    current,
                )
            )

            current = [
                annotation
            ]

        else:

            current.append(
                annotation
            )

    if current:

        result.append(
            (
                paragraph_number,
                current,
            )
        )

    return result


# ============================================================================
# Coherence-aware merging
# ============================================================================

def _merge_short_structural_blocks(
    document: LegalDocument,
    blocks: List[
        tuple[int | None, List[LegalAnnotation]]
    ],
    max_chars: int,
) -> List[
    tuple[int | None, List[LegalAnnotation]]
]:
    """
    Second-pass structural merge.

    This handles cases where OpenNyAI splits a legal paragraph into:

        heading
        paragraph

    or:

        enumeration marker
        enumeration content

    It does NOT merge arbitrary short legal propositions.
    """

    if not blocks:
        return []

    output: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    i = 0

    while i < len(blocks):

        paragraph_number, block = blocks[i]

        if not block:
            i += 1
            continue

        block_text = " ".join(
            annotation.text.strip()
            for annotation in block
        ).strip()

        should_attach = (
            _is_structural_only(
                block_text
            )
            or _is_heading_only(
                block_text
            )
        )

        if (
            should_attach
            and i + 1 < len(blocks)
        ):

            next_paragraph, next_block = (
                blocks[i + 1]
            )

            combined_start = (
                block[0].char_start
            )

            combined_end = (
                next_block[-1].char_end
            )

            combined_length = (
                combined_end
                - combined_start
            )

            if combined_length <= max_chars:

                output.append(
                    (
                        next_paragraph
                        if next_paragraph is not None
                        else paragraph_number,
                        [
                            *block,
                            *next_block,
                        ],
                    )
                )

                i += 2
                continue

        output.append(
            (
                paragraph_number,
                block,
            )
        )

        i += 1

    return output


# ============================================================================
# Main chunking
# ============================================================================

def chunk_document(
    document: LegalDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_standalone_chars: int = DEFAULT_MIN_STANDALONE_CHARS,
) -> List[LegalChunk]:
    """
    Chunk one LegalDocument into legal retrieval units.

    Main strategy
    -------------
    1. Numbered judgment paragraphs are primary boundaries.
    2. RRC role transitions do not automatically split paragraphs.
    3. Enumerated legal lists are split into logical items.
    4. Marker-only fragments attach to meaningful following text.
    5. Headings attach to their following legal text.
    6. Oversized blocks split only at RRC boundaries.
    7. Source text is always reconstructed from document.text.
    8. No arbitrary "merge every chunk under N chars" rule.
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be > 0."
        )

    if not document.annotations:
        return []

    # ------------------------------------------------------------------
    # 1. Build numbered paragraph blocks
    # ------------------------------------------------------------------

    paragraph_blocks = _build_paragraph_blocks(
        document
    )

    # ------------------------------------------------------------------
    # 2. Refine explicit enumerations
    # ------------------------------------------------------------------

    refined_blocks: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    for paragraph_number, block in paragraph_blocks:

        refined_blocks.extend(
            _split_enumerated_block(
                paragraph_number,
                block,
            )
        )

    # ------------------------------------------------------------------
    # 3. Repair marker-only / heading-only blocks
    # ------------------------------------------------------------------

    refined_blocks = _repair_structural_fragments(
        document,
        refined_blocks,
    )

    # ------------------------------------------------------------------
    # 4. Second structural merge
    # ------------------------------------------------------------------

    refined_blocks = _merge_short_structural_blocks(
        document,
        refined_blocks,
        max_chars,
    )

    # ------------------------------------------------------------------
    # 5. Split oversized blocks
    # ------------------------------------------------------------------

    final_blocks: List[
        tuple[int | None, List[LegalAnnotation]]
    ] = []

    for paragraph_number, block in refined_blocks:

        final_blocks.extend(
            _split_oversized_block(
                document,
                paragraph_number,
                block,
                max_chars,
            )
        )

    # ------------------------------------------------------------------
    # 6. Create chunks
    # ------------------------------------------------------------------

    chunks: List[LegalChunk] = []

    for _, block in final_blocks:

        if not block:
            continue

        chunk = _create_chunk(
            document=document,
            annotations=block,
            chunk_index=len(chunks),
        )

        chunks.append(
            chunk
        )

    # ------------------------------------------------------------------
    # 7. Final validation
    # ------------------------------------------------------------------

    validate_chunks(
        document,
        chunks,
    )

    return chunks


# ============================================================================
# Validation
# ============================================================================

def validate_chunks(
    document: LegalDocument,
    chunks: List[LegalChunk],
) -> None:
    """
    Validate chunk integrity.

    Checks:

        - indexes
        - document identity
        - offsets
        - source integrity
        - annotation counts
        - role metadata
    """

    previous_end = -1

    for expected_index, chunk in enumerate(chunks):

        if chunk.chunk_index != expected_index:

            raise ValueError(
                f"Chunk index mismatch: "
                f"expected {expected_index}, "
                f"got {chunk.chunk_index}."
            )

        if chunk.document_id != document.document_id:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"belongs to document "
                f"{chunk.document_id}, "
                f"expected {document.document_id}."
            )

        if chunk.char_start < 0:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"has negative char_start."
            )

        if chunk.char_end < chunk.char_start:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"has invalid offsets."
            )

        if chunk.char_end > document.text_length:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"extends beyond document."
            )

        if chunk.char_start < previous_end:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"overlaps previous chunk."
            )

        # --------------------------------------------------------------
        # Most important integrity check
        # --------------------------------------------------------------

        expected_text = document.text[
            chunk.char_start:chunk.char_end
        ]

        if chunk.text != expected_text:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"does not exactly match "
                f"authoritative source text."
            )

        if chunk.source_char_count != (
            chunk.char_end
            - chunk.char_start
        ):

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"source_char_count mismatch."
            )

        if chunk.text_char_count != len(
            chunk.text
        ):

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"text_char_count mismatch."
            )

        if chunk.annotation_count != len(
            chunk.annotation_ids
        ):

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"annotation_count mismatch."
            )

        if not chunk.roles:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"has no roles."
            )

        if chunk.primary_role not in chunk.roles:

            raise ValueError(
                f"Chunk {chunk.chunk_index} "
                f"primary_role is not present "
                f"in roles."
            )

        previous_end = chunk.char_end


# ============================================================================
# Serialization
# ============================================================================

def chunk_to_dict(
    chunk: LegalChunk,
) -> Dict:
    """Convert a LegalChunk to JSON-compatible dict."""

    return asdict(
        chunk
    )


def chunks_to_dict(
    chunks: List[LegalChunk],
) -> List[Dict]:
    """Convert chunks to JSON-compatible dictionaries."""

    return [
        chunk_to_dict(chunk)
        for chunk in chunks
    ]


# ============================================================================
# Statistics
# ============================================================================

def chunk_statistics(
    chunks: List[LegalChunk],
) -> Dict:
    """
    Generate chunking diagnostics.
    """

    if not chunks:

        return {
            "chunk_count": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_chars": 0,
            "tiny_chunks_lt_100": 0,
            "structural_fragments_lt_100": 0,
            "roles": {},
            "chunks_with_entities": 0,
            "chunks_with_summary_context": 0,
        }

    lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    role_counts: Dict[str, int] = {}

    for chunk in chunks:

        role = chunk.primary_role

        role_counts[role] = (
            role_counts.get(
                role,
                0,
            )
            + 1
        )

    tiny_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.text) < 100
    ]

    structural_fragments = [
        chunk
        for chunk in tiny_chunks
        if (
            _is_structural_only(
                chunk.text
            )
            or _is_heading_only(
                chunk.text
            )
        )
    ]

    return {
        "chunk_count": len(chunks),

        "min_chars": min(lengths),

        "max_chars": max(lengths),

        "avg_chars": round(
            sum(lengths)
            / len(lengths),
            2,
        ),

        "tiny_chunks_lt_100": len(
            tiny_chunks
        ),

        "structural_fragments_lt_100": len(
            structural_fragments
        ),

        "roles": role_counts,

        "chunks_with_entities": sum(
            bool(chunk.entities)
            for chunk in chunks
        ),

        "chunks_with_summary_context": sum(
            bool(chunk.summary_context)
            for chunk in chunks
        ),
    }


# ============================================================================
# Debug export
# ============================================================================

def export_chunk_debug(
    chunks: List[LegalChunk],
    path: str | Path,
) -> None:
    """
    Export detailed chunk information for manual legal audit.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "statistics": chunk_statistics(
            chunks
        ),

        "chunks": chunks_to_dict(
            chunks
        ),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================================
# Retrieval helpers
# ============================================================================

def is_argument_chunk(
    chunk: LegalChunk,
) -> bool:
    """Return True for party-argument chunks."""

    return bool(
        {
            "ARG_PETITIONER",
            "ARG_RESPONDENT",
        }
        & set(chunk.roles)
    )


def is_holding_chunk(
    chunk: LegalChunk,
) -> bool:
    """
    Return True for potentially dispositive / holding material.

    Note:
        RATIO is not the only holding signal.
    """

    return bool(
        {
            "RATIO",
            "RPC",
        }
        & set(chunk.roles)
    )


def is_fact_chunk(
    chunk: LegalChunk,
) -> bool:
    """Return True for factual/background chunks."""

    return bool(
        {
            "FAC",
            "RLC",
        }
        & set(chunk.roles)
    )


def is_precedent_chunk(
    chunk: LegalChunk,
) -> bool:
    """Return True for precedent-related chunks."""

    return bool(
        {
            "PRE_RELIED",
            "PRE_NOT_RELIED",
        }
        & set(chunk.roles)
    )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MIN_STANDALONE_CHARS",

    "LegalChunk",

    "chunk_document",

    "chunk_to_dict",
    "chunks_to_dict",

    "chunk_statistics",

    "validate_chunks",

    "export_chunk_debug",

    "is_argument_chunk",
    "is_holding_chunk",
    "is_fact_chunk",
    "is_precedent_chunk",
]