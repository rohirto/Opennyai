"""
Google Gemini implementation for the Legal RAG system.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from rag.generation.llm import (
    LegalLLM,
    LLMResponse,
)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class GeminiConfig:

    model_name: str = "gemini-3.5-flash"

    api_key: Optional[str] = None

    temperature: float = 0.1

    top_p: float = 0.9

    max_output_tokens: int = 1200

    timeout: int = 180


# =============================================================================
# GEMINI IMPLEMENTATION
# =============================================================================


class GeminiLegalLLM(LegalLLM):

    provider = "gemini"


    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
    ):

        self.config = config or GeminiConfig()

        api_key = (
            self.config.api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:

            raise RuntimeError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY environment variable."
            )


        self.client = genai.Client(
            api_key=api_key
        )


    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:

        try:

            self.client.models.get(
                model=self.config.model_name
            )

            return True

        except Exception:

            return False


    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:

        start_ns = time.perf_counter_ns()


        # Gemini allows system instructions separately from the user content.
        response = self.client.models.generate_content(

            model=self.config.model_name,

            contents=user_prompt,

            config=types.GenerateContentConfig(

                system_instruction=system_prompt,

                temperature=self.config.temperature,

                top_p=self.config.top_p,

                max_output_tokens=self.config.max_output_tokens,
            ),
        )


        duration_ns = time.perf_counter_ns() - start_ns


        text = getattr(
            response,
            "text",
            None,
        )


        if not text:

            text = ""


        # Gemini's SDK response structure can vary slightly between versions.
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )


        prompt_tokens = None
        completion_tokens = None


        if usage is not None:

            prompt_tokens = getattr(
                usage,
                "prompt_token_count",
                None,
            )

            completion_tokens = getattr(
                usage,
                "candidates_token_count",
                None,
            )


        return LLMResponse(

            text=text.strip(),

            model=self.config.model_name,

            prompt_tokens=prompt_tokens,

            completion_tokens=completion_tokens,

            total_duration_ns=duration_ns,

            provider="gemini",
        )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "GeminiConfig",
    "GeminiLegalLLM",
]