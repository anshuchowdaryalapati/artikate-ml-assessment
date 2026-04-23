"""PDF loader with page-level metadata for legal/document RAG."""
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text page-by-page from a PDF.
    
    Returns list of dicts: [{document, page, text}]
    Page numbers are 1-indexed to match human reading.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append({
                "document": pdf_path.name,
                "page": page_num,
                "text": text,
            })
    return pages


def load_all_pdfs(data_dir: str) -> List[Dict]:
    """Load every PDF in the data directory."""
    data_dir = Path(data_dir)
    all_pages = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")
    for pdf in pdf_files:
        print(f"Loading {pdf.name}...")
        pages = load_pdf(pdf)
        print(f"  {len(pages)} pages extracted")
        all_pages.extend(pages)
    return all_pages


if __name__ == "__main__":
    # Smoke test
    pages = load_all_pdfs("section2_rag/data")
    print(f"\nTotal: {len(pages)} pages from {len(set(p['document'] for p in pages))} documents")
    print(f"\nSample page 1 snippet:\n{pages[0]['text'][:200]}...")

