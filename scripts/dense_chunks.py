from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

tok = EmbeddingService().model.tokenizer
tok.model_max_length = 10_000  # warning band

chunks = chunk_pages(extract_pages(PDF), chunk_size=800, chunk_overlap=120)

rows = []
for c in chunks:
    n = len(tok.encode(c.text, add_special_tokens=False))
    rows.append((n, len(c.text) / n, c))

rows.sort(key=lambda r: r[0], reverse=True)  # max tokens first

for n, cpt, c in rows[:8]:
    print(f"page {c.metadata['page_number']}  tokens={n}  chars/token={cpt:.2f}")
    print(repr(c.text[:300]))
    print()
