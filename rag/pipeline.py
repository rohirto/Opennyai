from __future__ import annotations

from rag.models import (
    GeneratedAnswer,
)

from rag.retrieval.query_classifier import (
    LegalQueryClassifier,
)

from rag.retrieval.metadata_filter import (
    LegalMetadataFilterBuilder,
)

from rag.generation.context_builder import (
    assemble_context,
)

from rag.generation.generator import (
    LegalAnswerGenerator,
)


class LegalRAGPipeline:
    """
    End-to-end Legal RAG orchestration.

    Flow:

        Query
          ↓
        Query Classification
          ↓
        Metadata Filter
          ↓
        Vector Retrieval
          ↓
        Legal Reranking
          ↓
        Context Assembly
          ↓
        Legal Prompt Construction
          ↓
        Ollama / Qwen
          ↓
        Generated Answer + Sources
    """

    def __init__(
        self,
        embedder,
        retriever,
        reranker,
        llm,
        metadata_filter=None,
        query_classifier=None,
        answer_generator=None,
    ):

        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

        self.query_classifier = (
            query_classifier
            or LegalQueryClassifier()
        )

        self.metadata_filter = (
            metadata_filter
            or LegalMetadataFilterBuilder()
        )

        self.answer_generator = (
            answer_generator
            or LegalAnswerGenerator(
                llm=llm
            )
        )

    # ==================================================================
    # ASK
    # ==================================================================

    def ask(
        self,
        query: str,
        retrieve_k: int = 20,
        final_k: int = 8,
        max_context_chars: int = 12000,
    ) -> GeneratedAnswer:

        query = query.strip()

        if not query:
            raise ValueError(
                "query must be a non-empty string"
            )

        # --------------------------------------------------------------
        # 1. QUERY CLASSIFICATION
        # --------------------------------------------------------------

        analysis = (
            self.query_classifier.classify(
                query
            )
        )

        # --------------------------------------------------------------
        # 2. METADATA FILTER
        # --------------------------------------------------------------

        qdrant_filter = (
            self.metadata_filter.build(
                analysis
            )
        )

        # Current retriever implementation
        # expects the filter through QueryAnalysis.
        analysis._qdrant_filter = (
            qdrant_filter
        )

        # --------------------------------------------------------------
        # 3. VECTOR RETRIEVAL
        # --------------------------------------------------------------

        retrieved = (
            self.retriever.retrieve(
                query=query,
                query_analysis=analysis,
                limit=retrieve_k,
            )
        )

        # --------------------------------------------------------------
        # 4. LEGAL RERANKING
        # --------------------------------------------------------------

        reranked = (
            self.reranker.rerank(
                results=retrieved,
                query_analysis=analysis,
                top_k=final_k,
            )
        )

        # --------------------------------------------------------------
        # 5. CONTEXT ASSEMBLY
        # --------------------------------------------------------------

        context = assemble_context(
            results=reranked,
            query_analysis=analysis,
            max_context_chars=max_context_chars,
        )

        # --------------------------------------------------------------
        # 6. QWEN GENERATION
        # --------------------------------------------------------------

        answer = self.answer_generator.generate(
            query=query,
            query_analysis=analysis,
            context=context,
            results=reranked,
        )

        # --------------------------------------------------------------
        # 7. PIPELINE DEBUG INFORMATION
        # --------------------------------------------------------------

        answer.debug.update(
            {
                "query_type":
                    analysis.query_type,

                "preferred_roles":
                    analysis.preferred_roles,

                "excluded_roles":
                    analysis.excluded_roles,

                "keywords":
                    analysis.keywords,

                "court":
                    analysis.court,

                "case_name":
                    analysis.case_name,

                "citation":
                    analysis.citation,

                "retrieved_before_rerank":
                    len(retrieved),

                "reranked_count":
                    len(reranked),

                "context_chunks":
                    len(context.included_chunk_ids),

                "context_characters":
                    context.character_count,

                "context_sections":
                    context.sections,

                "context_debug":
                    context.debug,
                
                "llm_context": context.text,

                "qdrant_filter_applied":
                    qdrant_filter is not None,
            }
        )

        return answer