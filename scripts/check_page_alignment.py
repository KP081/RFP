from pathlib import Path
from app.services.document_parser import extract_pages

PDF = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")

EXPECTED = {
    14: "appendix-1 to tis",
    17: "qualification criteria",
    40: "dpr rating",
    78: "submission of final qap",
    96: "penalty for delay",
    113: "classified traffic volume count",
    151: "eight stages",
    282: "evaluation/ scoring criteria",
    353: "format of financial proposal",
}

pages = extract_pages(PDF)
print("Total pages:", len(pages))

for number, phrase in EXPECTED.items():
    page = next(p for p in pages if p.page_number == number)
    text = " ".join(page.text.split()).lower()
    print(number, "OK" if phrase in text else "MISMATCH")