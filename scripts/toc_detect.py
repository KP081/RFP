import re
from pathlib import Path

from app.services.document_parser import extract_pages

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")
DOT_LEADER = re.compile(r"\.{5,}")  # with 5+ dots

rows = []
for page in extract_pages(PDF):
    lines = [ln for ln in page.text.splitlines() if ln.strip()]
    if not lines:
        continue
    hits = sum(1 for ln in lines if DOT_LEADER.search(ln))
    rows.append((hits / len(lines), hits, len(lines), page.page_number))

rows.sort(reverse=True)

for ratio, hits, total, number in rows[:15]:
    print(f"page {number:>3}  dot-leader lines {hits:>3}/{total:<3}  ratio={ratio:.0%}")
