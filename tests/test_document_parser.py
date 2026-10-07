from pathlib import Path

from app.services.document_parser import extract_pages


RFP_FILE = Path("uploads/cef53ab4-84a8-407d-aef7-176f7926b306.pdf")


def test_extract_pdf_pages():
    pages = extract_pages(RFP_FILE)

    assert len(pages) > 0

    print(f"\nTotal pages: {len(pages)}")

    for page in pages[:5]:
        print("\n" + "=" * 80)
        print(f"Page: {page.page_number}")
        print(f"Characters: {len(page.text)}")
        print("=" * 80)
        print(page.text[:500])
