from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.document_parser import extract_pages
from app.services.embeddings import EmbeddingService

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

PROBES = [
    # (label, page, phrase (lowercase), role)
    ("Q15 cost", 353, "per km dpr cost", "failing"),
    ("Q3 turnover", 17, "crores", "failing"),
    ("Q9 marks", 282, "marking and evaluation scheme", "control"),
    ("Q10 key experts", 280, "man months input", "control"),
]

chunks = chunk_pages(extract_pages(PDF))
model = EmbeddingService().model
tok = model.tokenizer
limit = model.max_seq_length - 2  # [CLS] aur [SEP] 2 slots le lete hain

for label, page, phrase, role in PROBES:
    for c in chunks:
        if c.metadata["page_number"] != page:
            continue
        text = " ".join(c.text.split()).lower()  # whitespace normalize
        idx = text.find(phrase)
        if idx == -1:
            continue
        before = len(tok.encode(text[:idx], add_special_tokens=False))
        total = len(tok.encode(text, add_special_tokens=False))
        status = "VISIBLE" if before < limit else "TRUNCATED"
        print(
            f"{label} [{role}] page {page} chunk {c.chunk_id}: "
            f"phrase at token {before} of {total} -> {status}"
        )
