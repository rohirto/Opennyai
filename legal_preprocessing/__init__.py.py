# legal_preprocessing/__init__.py

from .legal_text_cleaner import (
    LegalTextCleaner,
    CleaningReport,
    clean_legal_text,
)

from .judgement_ingester import (
    LegalDocument,
    ingest_legal_document
)

__all__ = [
    "LegalTextCleaner",
    "CleaningReport",
    "clean_legal_text",
    "LegalDocument",
    "ingest_legal_document",
]