"""
Factory for creating the configured Legal RAG LLM.
"""

from __future__ import annotations

import config

from rag.generation.llm import (
    LegalLLM,
    OllamaConfig,
    OllamaLegalLLM,
)

from rag.generation.gemini_llm import (
    GeminiConfig,
    GeminiLegalLLM,
)


def create_llm() -> LegalLLM:

    provider = (
        getattr(
            config,
            "LLM_PROVIDER",
            "qwen",
        )
        or "qwen"
    ).strip().lower()


    # =========================================================================
    # QWEN / OLLAMA
    # =========================================================================

    if provider == "qwen":

        return OllamaLegalLLM(

            OllamaConfig(

                base_url=getattr(
                    config,
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434",
                ),

                model_name=getattr(
                    config,
                    "OLLAMA_MODEL",
                    "qwen2.5:3b-instruct",
                ),

                temperature=getattr(
                    config,
                    "LLM_TEMPERATURE",
                    0.1,
                ),

                top_p=getattr(
                    config,
                    "LLM_TOP_P",
                    0.9,
                ),

                num_predict=getattr(
                    config,
                    "LLM_MAX_OUTPUT_TOKENS",
                    1200,
                ),

                timeout=getattr(
                    config,
                    "OLLAMA_TIMEOUT",
                    180,
                ),

                keep_alive=getattr(
                    config,
                    "OLLAMA_KEEP_ALIVE",
                    "10m",
                ),
            )
        )


    # =========================================================================
    # GEMINI
    # =========================================================================

    if provider == "gemini":

        return GeminiLegalLLM(

            GeminiConfig(

                model_name=getattr(
                    config,
                    "GEMINI_MODEL",
                    "gemini-3.5-flash",
                ),

                api_key=getattr(
                    config,
                    "GEMINI_API_KEY",
                    None,
                ),

                temperature=getattr(
                    config,
                    "LLM_TEMPERATURE",
                    0.1,
                ),

                top_p=getattr(
                    config,
                    "LLM_TOP_P",
                    0.9,
                ),

                max_output_tokens=getattr(
                    config,
                    "LLM_MAX_OUTPUT_TOKENS",
                    1200,
                ),

                timeout=getattr(
                    config,
                    "GEMINI_TIMEOUT",
                    180,
                ),
            )
        )


    raise ValueError(

        f"Unsupported LLM_PROVIDER={provider!r}. "

        "Expected 'qwen' or 'gemini'."
    )


__all__ = [
    "create_llm",
]