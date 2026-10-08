from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

pages = extract_pages(PDF)
model = EmbeddingService().model
tok = model.tokenizer
limit = model.max_seq_length - 2
tok.model_max_length = 10_000  # for stop "256 > limit" warning

for size in [600, 800, 1000, 1200, 1500, 2000]:
    overlap = int(size * 0.15)
    chunks = chunk_pages(pages, chunk_size=size, chunk_overlap=overlap)
    counts = [len(tok.encode(c.text, add_special_tokens=False)) for c in chunks]

    over = [n for n in counts if n > limit]
    lost = sum(n - limit for n in over)

    print(
        f"size={size:>4} overlap={overlap:>3} chunks={len(chunks):>5} "
        f"over_limit={len(over) / len(counts):>4.0%} "
        f"lost_text={lost / sum(counts):>4.0%} "
        f"max_tokens={max(counts)}"
    )
