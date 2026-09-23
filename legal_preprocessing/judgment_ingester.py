from pathlib import Path
import pdfplumber

from .legal_text_cleaner import LegalTextCleaner


class LegalDocument:
    def __init__(
        self,
        pdf_path: Path,
        raw_text: str,
        cleaned_text: str,
        preprocessing_report,
    ):
        self.pdf_path = pdf_path
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text
        self.preprocessing_report = preprocessing_report


def extract_pdf_text(pdf_path: Path) -> str:
    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""

            if text:
                page_texts.append(text)

    return "\n\n".join(page_texts)


def ingest_legal_document(pdf_path: Path) -> LegalDocument:
    raw_text = extract_pdf_text(pdf_path)

    if not raw_text.strip():
        raise RuntimeError(
            f"PDF extraction returned empty text: {pdf_path}"
        )

    cleaner = LegalTextCleaner()

    cleaned_text, report = cleaner.clean(raw_text)

    return LegalDocument(
        pdf_path=pdf_path,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        preprocessing_report=report,
    )