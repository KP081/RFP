from dataclasses import dataclass

from app.services.document_parser import DocumentPage


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    metadata: dict


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            paragraph_break = text.rfind("\n\n", start, end)

            if paragraph_break > start + chunk_size // 2:
                end = paragraph_break

        chunk_text = text[start:end].strip()

        if chunk_text:
            actual_start = start
            actual_end = end

            chunks.append(
                (
                    chunk_text,
                    actual_start,
                    actual_end,
                )
            )

        if end >= text_length:
            break

        start = max(
            end - chunk_overlap,
            start + 1,
        )

    return chunks


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

        page_chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text, start, end in page_chunks:
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "page_number": page.page_number,
                        "start": start,
                        "end": end,
                    },
                )
            )

            chunk_id += 1

    return chunks
