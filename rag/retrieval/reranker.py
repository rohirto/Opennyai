from __future__ import annotations

from typing import List

from rag.models import RetrievalResult


class LegalReranker:

    ROLE_BONUS = {

        "decision": {
            "RATIO": 0.08,
            "RPC": 0.07,
            "ANALYSIS": 0.05,
            "PRE_RELIED": 0.02,
        },

        "precedent": {
            "PRE_RELIED": 0.08,
            "ANALYSIS": 0.06,
            "RATIO": 0.05,
        },

        "facts": {
            "FAC": 0.08,
            "RLC": 0.06,
            "ANALYSIS": 0.03,
        },

        "arguments": {
            "ARG_PETITIONER": 0.08,
            "ARG_RESPONDENT": 0.08,
            "ANALYSIS": 0.03,
        },

        "issue": {
            "ANALYSIS": 0.06,
            "RLC": 0.05,
            "RATIO": 0.04,
        },

        "general": {
            "ANALYSIS": 0.05,
            "RATIO": 0.05,
            "RPC": 0.04,
            "PRE_RELIED": 0.02,
        },
    }

    def rerank(
        self,
        results: List[RetrievalResult],
        query_analysis,
        top_k: int = 8,
    ) -> List[RetrievalResult]:

        role_bonus_map = self.ROLE_BONUS.get(
            query_analysis.query_type,
            self.ROLE_BONUS["general"],
        )

        preferred_roles = set(
            query_analysis.preferred_roles or []
        )

        for result in results:

            vector_score = (
                result.vector_score
                if result.vector_score is not None
                else result.score
            )

            primary_bonus = role_bonus_map.get(
                result.primary_role,
                0.0,
            )

            overlap = len(
                set(result.roles or [])
                & preferred_roles
            )

            role_overlap_bonus = min(
                overlap * 0.015,
                0.045,
            )

            rerank_score = (
                vector_score
                + primary_bonus
                + role_overlap_bonus
            )

            result.rerank_score = rerank_score

            result.retrieval_signals = {
                "vector_score": vector_score,
                "role_bonus": primary_bonus,
                "role_overlap_bonus": role_overlap_bonus,
            }

        results.sort(
            key=lambda x: (
                x.rerank_score
                if x.rerank_score is not None
                else x.vector_score or 0.0
            ),
            reverse=True,
        )

        return results[:top_k]