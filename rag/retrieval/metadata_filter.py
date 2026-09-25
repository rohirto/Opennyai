# rag/retrieval/metadata_filter.py

from __future__ import annotations

from typing import Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)


class LegalMetadataFilterBuilder:

    def build(
        self,
        analysis,
    ) -> Optional[Filter]:

        case_name = (
            getattr(
                analysis,
                "case_name",
                None,
            )
            or getattr(
                analysis,
                "case_query",
                None,
            )
        )

        citation = getattr(
            analysis,
            "citation",
            None,
        )

        court = getattr(
            analysis,
            "court",
            None,
        )

        # --------------------------------------------------------------
        # Citation is unambiguous -> exact filter.
        # --------------------------------------------------------------

        if citation:
            return Filter(
                must=[
                    FieldCondition(
                        key="citation",
                        match=MatchValue(
                            value=str(
                                citation
                            ).strip()
                        ),
                    )
                ]
            )

        # --------------------------------------------------------------
        # Case name.
        #
        # We intentionally do NOT require exact case_name here.
        # The retrieval layer will perform the case identity matching
        # against case_name/source_file/text.
        #
        # Therefore, return no hard filter when case_name is only a
        # natural-language name.
        # --------------------------------------------------------------

        if case_name:
            return None

        # --------------------------------------------------------------
        # Court is safe as a hard constraint only when explicitly given.
        # --------------------------------------------------------------

        if court:
            return Filter(
                must=[
                    FieldCondition(
                        key="court",
                        match=MatchValue(
                            value=str(
                                court
                            ).strip()
                        ),
                    )
                ]
            )

        return None