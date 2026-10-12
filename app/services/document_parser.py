from dataclasses import dataclass
from pathlib import Path

from docx.table import Table
from docx.text.paragraph import Paragraph
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

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=text.strip(),
            )
        )

    return pages


def _iter_docx_blocks(document):
    """Paragraphs or tables gives as a order"""
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "p":
            yield Paragraph(child, document).text.strip()

        elif tag == "tbl":
            table = Table(child, document)

            for row in table.rows:
                cells = []

                for cell in row.cells:
                    cell_text = cell.text.strip()

                    if cell_text and (not cells or cells[-1] != cell_text):
                        cells.append(cell_text)

                if cells:
                    yield " | ".join(cells)


def extract_pages_from_docx(file_path: Path) -> list[DocumentPage]:
    document = Document(file_path)

    blocks = [block for block in _iter_docx_blocks(document) if block]
    text = "\n\n".join(blocks)

    if not text:
        return []

    return [DocumentPage(page_number=1, text=text)]


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
