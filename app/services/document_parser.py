from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


@dataclass
class DocumentPage:
    page_number: int
    text: str


def extract_pages_from_pdf(file_path: Path) -> list[DocumentPage]:
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                DocumentPage(
                    page_number=page_number,
                    text=text.strip(),
                )
            )

    return pages


def extract_pages_from_docx(file_path: Path) -> list[DocumentPage]:
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)

    if not text:
        return []

    # DOCX does not have reliable page information available
    # through python-docx, so we treat the document as one page.
    return [
        DocumentPage(
            page_number=1,
            text=text,
        )
    ]


def extract_pages(file_path: Path) -> list[DocumentPage]:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pages_from_pdf(file_path)

    if extension == ".docx":
        return extract_pages_from_docx(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def extract_text(file_path: Path) -> str:
    pages = extract_pages(file_path)

    return "\n\n".join(page.text for page in pages)
