from dataclasses import dataclass

from app.services.document_parser import DocumentPage


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    metadata: dict


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 2000,
    chunk_overlap: int = 300,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Prefer ending at a paragraph boundary.
            if end < text_length:
                paragraph_break = text.rfind("\n\n", start, end)

                if paragraph_break > start + chunk_size // 2:
                    end = paragraph_break

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk,
                        metadata={
                            "page_number": page.page_number,
                            "start": start,
                            "end": end,
                        },
                    )
                )

                chunk_id += 1

            if end >= text_length:
                break

            start = max(end - chunk_overlap, start + 1)

    return chunks
