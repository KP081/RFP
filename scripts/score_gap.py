from pathlib import Path

import numpy as np

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService
from app.services.ingestion_service import CHUNK_OVERLAP, CHUNK_SIZE

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

# (question, expected pages, top-1 score jo analyze_failures ne dikhaya)
CASES = [
    (
        "What is the maximum marks distribution for the technical proposal evaluation?",
        [282],
        0.592,
    ),
    (
        "Which key experts are required and what is the man-month input of each?",
        [280, 355],
        0.548,
    ),
    (
        "What is the stage-wise payment schedule for the consultant as a percentage of contract price?",
        [78, 79],
        0.778,
    ),
]

embedder = EmbeddingService()
chunks = chunk_pages(
    extract_pages(PDF), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

for question, pages, top1 in CASES:
    q = np.array(embedder.embed_text(question))
    print("=" * 80)
    print(question)
    print(f"top-1 score in retrieval: {top1}")

    for c in chunks:
        if c.metadata["page_number"] not in pages:
            continue
        v = np.array(embedder.embed_text(c.text))

        print(
            f"  page {c.metadata['page_number']} chunk {c.chunk_id} "
            f"score={float(q @ v):.3f}"
        )
        print("    ", repr(c.text[:160]))
