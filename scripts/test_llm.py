from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rag.generation.llm_factory import create_llm


def main():

    llm = create_llm()

    print("=" * 80)
    print("Provider :", llm.provider)
    print("Testing LLM with legal evidence...")
    print("=" * 80)

    legal_evidence = """
[Evidence Role: formal disposition]
[Chunk: 67]
[Role: RPC]

45. All the appeals which we have heard simultaneously are allowed
in the above terms and the judgments impugned are modified accordingly.
The writ petitions brought by employees or their representatives shall
also stand disposed of in the same terms.

[Evidence Role: formal disposition]
[Chunk: 68]
[Role: RPC]

46. Pending application(s), if any, shall also stand disposed of.
"""

    response = llm.generate(

        system_prompt=(
            "You are a legal research assistant. "
            "Answer the question using ONLY the supplied legal evidence. "
            "Do not use outside knowledge. "
            "Cite the relevant chunk using [Chunk N]."
        ),

        user_prompt=f"""
LEGAL QUESTION
==============

Were the appeals allowed or dismissed?


LEGAL EVIDENCE
==============

{legal_evidence}


ANSWER
======

Answer directly and concisely.
""",
    )

    print()
    print("MODEL:")
    print(response.model)

    print()
    print("PROVIDER:")
    print(response.provider)

    print()
    print("ANSWER:")
    print(response.text)

    print()
    print("PROMPT TOKENS:")
    print(response.prompt_tokens)

    print()
    print("COMPLETION TOKENS:")
    print(response.completion_tokens)


if __name__ == "__main__":
    main()