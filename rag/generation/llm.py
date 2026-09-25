"""
LLM abstraction for the Legal RAG system.

Supported providers:
    - Ollama / Qwen
    - Gemini

The rest of the RAG pipeline should depend only on LegalLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests


# =============================================================================
# COMMON RESPONSE
# =============================================================================


@dataclass
class LLMResponse:
    text: str
    model: str

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_duration_ns: Optional[int] = None

    provider: Optional[str] = None


# =============================================================================
# COMMON INTERFACE
# =============================================================================


class LegalLLM:
    """
    Provider-independent interface for legal answer generation.
    """

    provider: str = "unknown"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        raise NotImplementedError


# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================


@dataclass
class OllamaConfig:

    base_url: str = "http://localhost:11434"

    model_name: str = "qwen2.5:3b-instruct"

    temperature: float = 0.1

    top_p: float = 0.9

    num_predict: int = 1200

    timeout: int = 180

    keep_alive: str = "10m"


# =============================================================================
# OLLAMA IMPLEMENTATION
# =============================================================================


class OllamaLegalLLM(LegalLLM):

    provider = "qwen"

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
    ):

        self.config = config or OllamaConfig()

        self.base_url = self.config.base_url.rstrip("/")


    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:

        try:

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            return True

        except requests.RequestException:

            return False


    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    def list_models(self):

        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()

        return payload.get(
            "models",
            [],
        )


    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:

        payload = {

            "model": self.config.model_name,

            "system": system_prompt,

            "prompt": user_prompt,

            "stream": False,

            "keep_alive": self.config.keep_alive,

            "options": {

                "temperature": self.config.temperature,

                "top_p": self.config.top_p,

                "num_predict": self.config.num_predict,
            },
        }


        response = requests.post(

            f"{self.base_url}/api/generate",

            json=payload,

            timeout=self.config.timeout,
        )

        response.raise_for_status()

        data = response.json()


        return LLMResponse(

            text=data.get(
                "response",
                "",
            ).strip(),

            model=data.get(
                "model",
                self.config.model_name,
            ),

            prompt_tokens=data.get(
                "prompt_eval_count",
            ),

            completion_tokens=data.get(
                "eval_count",
            ),

            total_duration_ns=data.get(
                "total_duration",
            ),

            provider="qwen",
        )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "LLMResponse",
    "LegalLLM",
    "OllamaConfig",
    "OllamaLegalLLM",
]