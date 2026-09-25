# rag/ingestion/embedder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"
    batch_size: int = 16
    normalize_embeddings: bool = True
    show_progress_bar: bool = True


class LegalEmbedder:
    """
    Embedding wrapper for legal RAG.

    Important:
    - Embeddings are generated from chunk.embedding_text.
    - Legal metadata such as roles/entities is NOT dumped blindly into
      the embedding text.
    - Metadata remains available for Qdrant filtering/reranking.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
    ):
        self.config = config or EmbeddingConfig()

        self.device = self._resolve_device(self.config.device)

        print("=" * 80)
        print("LEGAL EMBEDDER")
        print("=" * 80)
        print(f"Model  : {self.config.model_name}")
        print(f"Device : {self.device}")

        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.device,
        )

        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"Dimension: {self.dimension}")
        print("=" * 80)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

        except Exception:
            pass

        return "cpu"

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
        normalize: bool | None = None,
        show_progress_bar: bool | None = None,
    ) -> np.ndarray:
        """
        Encode one or more texts.

        Returns:
            np.ndarray of shape:
                (n_texts, embedding_dimension)
        """

        if isinstance(texts, str):
            texts = [texts]

        texts = list(texts)

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.config.batch_size,
            normalize_embeddings=(
                self.config.normalize_embeddings
                if normalize is None
                else normalize
            ),
            show_progress_bar=(
                self.config.show_progress_bar
                if show_progress_bar is None
                else show_progress_bar
            ),
            convert_to_numpy=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a user query.
        """

        embedding = self.encode(
            [query],
            show_progress_bar=False,
        )

        return embedding[0]

    def encode_chunks(
        self,
        chunks,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Encode LegalChunk objects.

        Uses chunk.embedding_text if available.
        Falls back to chunk.text.
        """

        texts = []

        for chunk in chunks:
            embedding_text = getattr(
                chunk,
                "embedding_text",
                None,
            )

            if embedding_text:
                texts.append(embedding_text)
            else:
                texts.append(chunk.text)

        return self.encode(
            texts,
            batch_size=batch_size,
        )

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Cosine similarity.

        Assumes embeddings are normalized.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        document_embeddings = np.asarray(
            document_embeddings,
            dtype=np.float32,
        )

        return document_embeddings @ query_embedding