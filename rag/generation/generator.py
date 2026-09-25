"""
Legal RAG answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rag.models import (
    GeneratedAnswer,
    QueryAnalysis,
    RetrievalResult,
    SourceCitation,
)

from rag.generation.context_builder import (
    AssembledContext,
)

from rag.generation.prompt import (
    SYSTEM_PROMPT,
    build_legal_prompt,
)

from rag.generation.llm import (
    LegalLLM,
)

from rag.generation.llm_factory import (
    create_llm,
)


# =============================================================================
# GENERATOR
# =============================================================================


class LegalAnswerGenerator:

    def __init__(
        self,
        llm: Optional[LegalLLM] = None,
    ):

        self.llm = llm or create_llm()


    # -------------------------------------------------------------------------
    # Source extraction
    # -------------------------------------------------------------------------

    def _build_sources(
        self,
        results: List[RetrievalResult],
    ) -> List[SourceCitation]:

        sources = []

        seen = set()


        for result in results:

            key = (
                result.document_id,
                result.chunk_index,
            )


            if key in seen:
                continue


            seen.add(key)


            sources.append(

                SourceCitation(

                    source_id=result.chunk_id,

                    document_id=result.document_id,

                    chunk_index=result.chunk_index,

                    source_file=result.source_file,

                    case_name=result.case_name,

                    citation=result.citation,

                    court=result.court,

                    judgment_date=result.judgment_date,

                    char_start=result.char_start,

                    char_end=result.char_end,

                    role=result.primary_role,
                )
            )


        return sources


    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    def generate(

        self,

        query: str,

        query_analysis: QueryAnalysis,

        context: AssembledContext,

        results: List[RetrievalResult],

    ) -> GeneratedAnswer:


        prompt = build_legal_prompt(

            query=query,

            query_analysis=query_analysis,

            context=context,
        )


        response = self.llm.generate(

            system_prompt=SYSTEM_PROMPT,

            user_prompt=prompt,
        )


        sources = self._build_sources(
            results
        )


        return GeneratedAnswer(

            answer=response.text,

            sources=sources,

            query=query,

            model=response.model,

            retrieval_count=len(
                results
            ),

            debug={

                "llm_provider": response.provider,

                "llm_model": response.model,

                "prompt_tokens": response.prompt_tokens,

                "completion_tokens": response.completion_tokens,

                "total_duration_ns": response.total_duration_ns,

                "context_characters": context.character_count,

                "context_chunks": len(
                    context.included_chunk_ids
                ),

            },
        )


__all__ = [
    "LegalAnswerGenerator",
]