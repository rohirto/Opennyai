# rag/retrieval/retriever.py

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from rag.models import RetrievalResult


class LegalRetriever:

    def __init__(
        self,
        qdrant_store,
        embedder,
    ):

        self.store = qdrant_store
        self.embedder = embedder

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize(
        value,
    ) -> str:

        if value is None:
            return ""

        value = str(value).casefold()

        value = re.sub(
            r"[_\-]+",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # ==================================================================
    # TOKENIZATION
    # ==================================================================

    @staticmethod
    def _tokens(
        value: str,
    ) -> set[str]:

        return {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                value.casefold(),
            )
            if len(token) >= 3
        }

    # ==================================================================
    # CASE IDENTITY SCORE
    # ==================================================================

    def _case_identity_score(
        self,
        result: RetrievalResult,
        case_name: Optional[str],
    ) -> float:
        """
        Calculate how strongly a result belongs to the requested case.

        Signals, in descending order:

            1. Exact case-name match in metadata
            2. Exact case-name match in source filename
            3. Exact case-name match in chunk text
            4. Token overlap with metadata/source
            5. Token overlap with chunk text

        This is a retrieval/ranking signal only.
        It does NOT discard results.
        """

        if not case_name:
            return 0.0

        target = self._normalize(
            case_name
        )

        if not target:
            return 0.0

        payload = (
            result.metadata
            or {}
        )

        metadata_case_name = self._normalize(
            payload.get(
                "case_name"
            )
            or result.case_name
        )

        source_file = self._normalize(
            payload.get(
                "source_file"
            )
            or result.source_file
        )

        citation = self._normalize(
            payload.get(
                "citation"
            )
            or result.citation
        )

        text = self._normalize(
            result.text
        )

        # --------------------------------------------------------------
        # Exact metadata match
        # --------------------------------------------------------------

        if (
            target
            and target == metadata_case_name
        ):
            return 1.00

        # --------------------------------------------------------------
        # Case name contained in metadata
        #
        # Example:
        # query      = "sunil kumar"
        # metadata   = "sunil kumar b"
        # --------------------------------------------------------------

        if (
            target
            and target in metadata_case_name
        ):
            return 0.95

        # --------------------------------------------------------------
        # Source filename match
        #
        # Example:
        # query      = "sunil kumar"
        # source     = "sunil kumar b.pdf"
        # --------------------------------------------------------------

        if (
            target
            and target in source_file
        ):
            return 0.90

        # --------------------------------------------------------------
        # Citation may identify the case
        # --------------------------------------------------------------

        if (
            target
            and target in citation
        ):
            return 0.85

        # --------------------------------------------------------------
        # Exact phrase in chunk text
        # --------------------------------------------------------------

        if (
            target
            and target in text
        ):
            return 0.75

        # --------------------------------------------------------------
        # Token overlap
        # --------------------------------------------------------------

        target_tokens = self._tokens(
            target
        )

        if not target_tokens:
            return 0.0

        metadata_tokens = (
            self._tokens(
                metadata_case_name
                + " "
                + source_file
                + " "
                + citation
            )
        )

        text_tokens = self._tokens(
            text
        )

        metadata_overlap = (
            len(
                target_tokens
                & metadata_tokens
            )
            / len(target_tokens)
        )

        text_overlap = (
            len(
                target_tokens
                & text_tokens
            )
            / len(target_tokens)
        )

        return max(
            0.60 * metadata_overlap,
            0.45 * text_overlap,
        )

    # ==================================================================
    # CASE QUERY EXTRACTION
    # ==================================================================

    @staticmethod
    def _get_case_name(
        query_analysis,
    ) -> Optional[str]:

        if query_analysis is None:
            return None

        case_name = getattr(
            query_analysis,
            "case_name",
            None,
        )

        if case_name:
            return str(
                case_name
            ).strip()

        case_query = getattr(
            query_analysis,
            "case_query",
            None,
        )

        if case_query:
            return str(
                case_query
            ).strip()

        return None

    # ==================================================================
    # RETRIEVE
    # ==================================================================

    def retrieve(
        self,
        query: str,
        query_analysis=None,
        query_filter=None,
        limit: int = 20,
    ) -> List[RetrievalResult]:

        # --------------------------------------------------------------
        # Encode query
        # --------------------------------------------------------------

        query_vector = self.embedder.encode_query(
            query
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Resolve Qdrant filter
        #
        # Explicit query_filter wins.
        #
        # Otherwise use the filter attached by LegalRAGPipeline:
        #
        #     analysis._qdrant_filter
        # --------------------------------------------------------------

        if query_filter is None:
            query_filter = getattr(
                query_analysis,
                "_qdrant_filter",
                None,
            )

        # --------------------------------------------------------------
        # QDRANT SEARCH
        # --------------------------------------------------------------

        try:

            response = self.store.client.query_points(
                collection_name=self.store.collection_name,
                query=query_vector.tolist(),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            points = response.points

        except AttributeError:

            points = self.store.client.search(
                collection_name=self.store.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        # --------------------------------------------------------------
        # Convert Qdrant points
        # --------------------------------------------------------------

        results = []

        for point in points:

            payload = point.payload or {}

            vector_score = float(
                point.score
            )

            result = RetrievalResult(

                chunk_id=str(
                    point.id
                ),

                document_id=payload.get(
                    "document_id",
                    "",
                ),

                chunk_index=int(
                    payload.get(
                        "chunk_index",
                        -1,
                    )
                ),

                text=payload.get(
                    "text",
                    "",
                ),

                score=vector_score,

                primary_role=payload.get(
                    "primary_role",
                    "UNKNOWN",
                ),

                roles=payload.get(
                    "roles",
                    [],
                ),

                char_start=int(
                    payload.get(
                        "char_start",
                        0,
                    )
                ),

                char_end=int(
                    payload.get(
                        "char_end",
                        0,
                    )
                ),

                source_file=payload.get(
                    "source_file",
                ),

                case_name=payload.get(
                    "case_name",
                ),

                citation=payload.get(
                    "citation",
                ),

                court=payload.get(
                    "court",
                ),

                judgment_date=payload.get(
                    "judgment_date",
                ),

                entities=payload.get(
                    "entities",
                    {},
                ),

                summary_context=payload.get(
                    "summary_context",
                    {},
                ),

                vector_score=vector_score,

                metadata=payload,
            )

            results.append(
                result
            )

        # --------------------------------------------------------------
        # CASE-AWARE RETRIEVAL
        # --------------------------------------------------------------

        case_name = self._get_case_name(
            query_analysis
        )

        if case_name:

            for result in results:

                identity_score = (
                    self._case_identity_score(
                        result,
                        case_name,
                    )
                )

                # Store debug values in payload.
                #
                # These are internal retrieval signals and will not
                # affect the public RetrievalResult schema.
                result.metadata[
                    "_case_identity_score"
                ] = identity_score

                result.metadata[
                    "_query_aware_score"
                ] = (
                    result.vector_score
                    + (
                        0.30
                        * identity_score
                    )
                )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # We do NOT completely replace semantic similarity.
            #
            # Case identity is an additional signal.
            # ----------------------------------------------------------

            results.sort(
                key=lambda result: (
                    -float(
                        result.metadata.get(
                            "_query_aware_score",
                            result.vector_score,
                        )
                    ),
                    -float(
                        result.vector_score
                    ),
                )
            )

        return results