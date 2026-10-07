from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages


RFP_FILE = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")


def test_rfp_chunking():
    pages = extract_pages(RFP_FILE)

    chunks = chunk_pages(
        pages,
        chunk_size=2000,
        chunk_overlap=300,
    )

    assert len(pages) == 379
    assert len(chunks) > 0

    print(f"\nTotal pages: {len(pages):,}")
    print(f"Total chunks: {len(chunks):,}")

    for chunk in chunks[:10]:
        print("\n" + "=" * 80)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.metadata['page_number']}")
        print(f"Characters: {len(chunk.text)}")
        print("=" * 80)
        print(chunk.text[:500])


def test_chunk_overlap():
    pages = [
        type(
            "Page",
            (),
            {
                "page_number": 1,
                "text": "A" * 5000,
            },
        )()
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) > 1
    assert chunks[0].text[-200:] == chunks[1].text[:200]
