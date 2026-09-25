"""
Legal RAG prompt construction.

The prompt is deliberately designed for legal question answering:

- answer only from supplied evidence
- distinguish present court from lower courts
- distinguish holdings from arguments
- preserve uncertainty
- cite supplied chunks
- do not invent authorities, provisions, dates, or conclusions
"""

from __future__ import annotations

from rag.models import QueryAnalysis
from rag.generation.context_builder import AssembledContext


SYSTEM_PROMPT = """
You are a legal research assistant for officers dealing with Indian
legal and statutory material.

Your task is to answer questions using ONLY the legal evidence supplied
in the CONTEXT.

IMPORTANT SOURCE-OF-TRUTH RULE:

The supplied legal evidence is the only source you may use for the answer.
Do not use your general knowledge to fill gaps.

You must follow these rules:

1. Do not invent:
   - cases
   - citations
   - statutory provisions
   - dates
   - facts
   - holdings
   - court directions
   - quotations
   - procedural history

2. Distinguish carefully between:
   - the decision of the present court,
   - decisions of lower courts,
   - arguments made by parties,
   - judicial reasoning,
   - cited precedents.

3. RPC / formal disposition evidence represents the present court's
   ruling and should receive the highest authority when answering what
   the court finally decided.

4. Do not present a lower court's decision as the present court's decision.

5. Do not present a party's argument as the court's holding.

6. For a decision or case-lookup question, answer in this order:
   a. final disposition;
   b. principal operative directions;
   c. important substantive holdings;
   d. supporting judicial reasoning.

7. If the evidence explicitly states the final disposition, state it
   directly. Do NOT abstain merely because other details are incomplete.

8. Abstain only when the supplied evidence genuinely does not establish
   the proposition requested by the user.

9. If evidence contains different outcomes from different courts,
   explicitly distinguish the courts instead of silently choosing one.

10. Every material proposition must have a citation in the form:
    [Chunk N]

11. A citation must support the proposition immediately preceding it.
    Do not cite chunks merely because they concern the same subject.

12. Do not reproduce long passages verbatim. Paraphrase the evidence.

13. Keep the answer concise but legally precise.

14. Do not provide legal advice or recommendations unless the supplied
    material itself contains such a recommendation.

Return a direct answer to the user's question.
"""


def build_legal_prompt(
    query: str,
    query_analysis: QueryAnalysis,
    context: AssembledContext,
) -> str:

    query_type = (
        query_analysis.query_type
        or "general"
    )

    preferred_roles = ", ".join(
        query_analysis.preferred_roles
    ) or "None specified"

    context_text = context.text.strip()

    if not context_text:
        context_text = (
            "[No legal evidence was retrieved.]"
        )

    # Case name is useful to the LLM as an explicit retrieval target.
    case_name = (
        getattr(
            query_analysis,
            "case_name",
            None,
        )
        or "None identified"
    )

    citation = (
        getattr(
            query_analysis,
            "citation",
            None,
        )
        or "None identified"
    )

    return f"""
LEGAL QUESTION
==============

{query}


QUERY TYPE
==========

{query_type}


IDENTIFIED CASE
===============

{case_name}


IDENTIFIED CITATION
===================

{citation}


PREFERRED LEGAL ROLES
=====================

{preferred_roles}


LEGAL EVIDENCE
==============

{context_text}


INSTRUCTIONS FOR THIS ANSWER
============================

Answer the legal question using ONLY the LEGAL EVIDENCE above.

For a decision or case-lookup question:

1. Identify the PRESENT COURT'S final decision first.

2. If the evidence explicitly states that an appeal, petition or
   application was allowed, dismissed, disposed of, partly allowed,
   modified, etc., state that disposition directly.

3. State the principal operative directions supported by the evidence.

4. State the important substantive holdings necessary to understand
   the decision.

5. Use supporting judicial reasoning only after the final decision and
   operative directions have been stated.

6. Do not confuse a lower-court order with the present court's decision.

7. Do not confuse a party's argument with the court's holding.

8. Cite every material proposition using [Chunk N].

9. If the final disposition is present in the evidence, DO NOT say that
   the context is insufficient merely because some other detail is
   missing.

10. Say the following only when the requested proposition genuinely
    cannot be established from the supplied evidence:

"The supplied context does not contain sufficient evidence to determine
this reliably."

Do not use general knowledge to complete the answer.
""".strip()


__all__ = [
    "SYSTEM_PROMPT",
    "build_legal_prompt",
]