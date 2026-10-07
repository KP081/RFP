from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

chunks = chunk_pages(extract_pages(PDF))   

model = EmbeddingService().model
limit = model.max_seq_length               # MiniLM: 256
tokenizer = model.tokenizer

counts = [
    len(tokenizer.encode(c.text, add_special_tokens=True, truncation=False))
    for c in chunks
]

over = [n for n in counts if n > limit]
lost = sum(n - limit for n in over)

print("max_seq_length:", limit)
print("chunks:", len(counts))
print(f"avg tokens/chunk: {sum(counts) / len(counts):.0f}")
print(f"chunks over limit: {len(over)} ({len(over) / len(counts):.0%})")
print(f"tokens never embedded: {lost / sum(counts):.0%} of all text")