# rag/ingestion/qdrant_ingest.py

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


DEFAULT_COLLECTION = "legal_documents"


class QdrantLegalStore:
    """
    Qdrant storage layer for the legal RAG corpus.

    Responsibilities
    ----------------
    - Connect to Qdrant
    - Create/manage the legal document collection
    - Create payload indexes
    - Convert LegalChunk objects to Qdrant payloads
    - Delete previous document versions
    - Validate embeddings
    - Upsert chunks in batches
    - Support vector search with optional metadata filtering
    """

    FILTER_FIELDS = {
        "document_id": PayloadSchemaType.KEYWORD,
        "source_file": PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "case_name": PayloadSchemaType.KEYWORD,
        "citation": PayloadSchemaType.KEYWORD,
        "court": PayloadSchemaType.KEYWORD,
        "primary_role": PayloadSchemaType.KEYWORD,
    }

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.embedding_model = embedding_model

        self.client = QdrantClient(url=url)

    # ==================================================================
    # HEALTH
    # ==================================================================

    def health_check(self) -> bool:
        try:
            self.client.get_collections()

            print(
                f"[Qdrant] Connected: {self.url}"
            )

            return True

        except Exception as exc:
            print(
                f"[Qdrant] Connection failed: {exc}"
            )

            return False

    # ==================================================================
    # COLLECTION
    # ==================================================================

    def create_collection(
        self,
        recreate: bool = False,
    ) -> None:

        exists = self.client.collection_exists(
            self.collection_name
        )

        if exists and recreate:

            print(
                f"[Qdrant] Deleting existing collection: "
                f"{self.collection_name}"
            )

            self.client.delete_collection(
                collection_name=self.collection_name
            )

            exists = False

        if not exists:

            print(
                f"[Qdrant] Creating collection: "
                f"{self.collection_name}"
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

        else:

            print(
                f"[Qdrant] Collection already exists: "
                f"{self.collection_name}"
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """
        Create indexes for fields used for exact legal-document filtering.
        """

        for field_name, field_type in self.FILTER_FIELDS.items():

            try:

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=True,
                )

                print(
                    f"[Qdrant] Payload index ready: "
                    f"{field_name}"
                )

            except Exception as exc:

                # Existing indexes or client/server version differences
                # should not abort the entire ingestion process.
                print(
                    f"[Qdrant] Payload index skipped "
                    f"({field_name}): {exc}"
                )

    # ==================================================================
    # POINT ID
    # ==================================================================

    @staticmethod
    def make_point_id(
        document_id: str,
        chunk_index: int,
    ) -> str:

        value = (
            f"{document_id}:chunk:{chunk_index}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                value,
            )
        )

    # ==================================================================
    # DOCUMENT CLEANUP
    # ==================================================================

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete all chunks belonging to one document.

        This prevents stale chunks from surviving when a document
        is reprocessed with fewer chunks.
        """

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
            wait=True,
        )

        print(
            f"[Qdrant] Deleted existing chunks for "
            f"document_id={document_id}"
        )

    # ==================================================================
    # PAYLOAD
    # ==================================================================

    def chunk_to_payload(
        self,
        chunk,
        document=None,
    ) -> dict:
        """
        Convert a LegalChunk into a Qdrant payload.

        Document-level metadata takes precedence over chunk-level
        metadata when a canonical LegalDocument is supplied.
        """

        def value_from_document_or_chunk(
            name: str,
            default=None,
        ):

            if document is not None:

                value = getattr(
                    document,
                    name,
                    None,
                )

                if value is not None:
                    return value

            return getattr(
                chunk,
                name,
                default,
            )

        case_name = value_from_document_or_chunk(
            "case_name"
        )

        return {

            # ----------------------------------------------------------
            # IDENTITY
            # ----------------------------------------------------------

            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,

            # ----------------------------------------------------------
            # TEXT
            # ----------------------------------------------------------

            "text": chunk.text,

            # ----------------------------------------------------------
            # LEGAL STRUCTURE
            # ----------------------------------------------------------

            "primary_role": chunk.primary_role,
            "roles": chunk.roles,
            "role_sequence": chunk.role_sequence,

            "annotation_ids": chunk.annotation_ids,

            # ----------------------------------------------------------
            # SOURCE LOCATION
            # ----------------------------------------------------------

            "char_start": chunk.char_start,
            "char_end": chunk.char_end,

            "source_char_count": chunk.source_char_count,
            "text_char_count": chunk.text_char_count,

            "annotation_count": chunk.annotation_count,

            # ----------------------------------------------------------
            # ENTITIES
            # ----------------------------------------------------------

            "entities": chunk.entities,
            "entity_ids": chunk.entity_ids,

            # ----------------------------------------------------------
            # SUMMARY
            # ----------------------------------------------------------

            "summary_context": chunk.summary_context,

            # ----------------------------------------------------------
            # DOCUMENT METADATA
            # ----------------------------------------------------------

            "source_file": value_from_document_or_chunk(
                "source_file"
            ),

            "source_type": value_from_document_or_chunk(
                "source_type"
            ),

            "language": value_from_document_or_chunk(
                "language"
            ),

            "case_name": case_name,

            "case_name_normalized": (
                str(case_name or "")
                .casefold()
                .strip()
            ),

            "case_aliases": self._build_case_aliases(
                case_name
            ),

            "citation": value_from_document_or_chunk(
                "citation"
            ),

            "court": value_from_document_or_chunk(
                "court"
            ),

            "judgment_date": value_from_document_or_chunk(
                "judgment_date"
            ),

            "schema_version": value_from_document_or_chunk(
                "schema_version"
            ),

            # ----------------------------------------------------------
            # EMBEDDING
            # ----------------------------------------------------------

            "embedding_model": self.embedding_model,
        }

    # ==================================================================
    # VECTOR VALIDATION
    # ==================================================================

    def validate_vector(
        self,
        vector,
    ) -> None:

        if len(vector) != self.vector_size:

            raise ValueError(
                f"Invalid vector dimension: "
                f"expected {self.vector_size}, "
                f"got {len(vector)}"
            )

        for value in vector:

            if not math.isfinite(float(value)):

                raise ValueError(
                    "Embedding contains NaN or infinity."
                )

    # ==================================================================
    # INGEST
    # ==================================================================

    def ingest(
        self,
        chunks,
        vectors,
        batch_size: int = 64,
        document=None,
        replace_document: bool = True,
    ) -> None:

        if len(chunks) != len(vectors):

            raise ValueError(
                "Number of chunks and vectors must match."
            )

        total = len(chunks)

        if total == 0:

            print(
                "[Qdrant] Nothing to ingest."
            )

            return

        # --------------------------------------------------------------
        # Delete previous version of this document
        # --------------------------------------------------------------

        if replace_document and document is not None:

            self.delete_document(
                document.document_id
            )

        # --------------------------------------------------------------
        # Upsert
        # --------------------------------------------------------------

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            points = []

            for chunk, vector in zip(
                chunks[start:end],
                vectors[start:end],
            ):

                self.validate_vector(
                    vector
                )

                point_id = self.make_point_id(
                    chunk.document_id,
                    chunk.chunk_index,
                )

                payload = self.chunk_to_payload(
                    chunk,
                    document=document,
                )

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist()
                        if hasattr(vector, "tolist")
                        else vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            print(
                f"[Qdrant] Indexed "
                f"{end}/{total}"
            )

    # ==================================================================
    # INFO
    # ==================================================================

    def info(self):
        return self.client.get_collection(
            self.collection_name
        )

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        query_vector,
        limit: int = 20,
        query_filter=None,
    ):
        """
        Vector search with optional Qdrant payload filter.
        """

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return response.points

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    @staticmethod
    def load_chunks(
        chunks_file: str | Path,
    ) -> list[dict]:

        path = Path(chunks_file)

        if not path.exists():

            raise FileNotFoundError(
                f"Chunks file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(handle)

        if not isinstance(data, list):

            raise ValueError(
                f"Expected a list in {path}"
            )

        return data

    # ==================================================================
    # CASE ALIASES
    # ==================================================================

    @staticmethod
    def _build_case_aliases(
        case_name,
    ) -> list[str]:

        if not case_name:
            return []

        value = (
            str(case_name)
            .casefold()
            .strip()
        )

        aliases = {
            value
        }

        # Remove common legal suffixes.
        for suffix in (
            " case",
            " judgment",
            " judgement",
        ):

            if value.endswith(suffix):

                aliases.add(
                    value[
                        :-len(suffix)
                    ].strip()
                )

        return sorted(
            aliases
        )