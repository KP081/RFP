from pathlib import Path
from app.services.document_parser import extract_pages

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")
pages = extract_pages(PDF)

for number in [14, 78, 282]:
    text = next(p.text for p in pages if p.page_number == number)
    print("=" * 80)
    print(f"page {number}: {len(text)} chars")
    print(r"count of \n\n :", text.count("\n\n"))
    print(r"count of \n   :", text.count("\n"))
    print(repr(text[:600]))     