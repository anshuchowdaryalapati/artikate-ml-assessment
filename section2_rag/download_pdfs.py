import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PDFS = [
    # GDPR full regulation - real legal document, ~100 pages
    ("https://gdpr-info.eu/wp-content/uploads/2019/04/gdpr-art.pdf", "gdpr_regulation.pdf"),
    # Sample NDA from public template repository
    ("https://www.uspto.gov/sites/default/files/documents/Sample%20NDA.pdf", "sample_nda.pdf"),
    # Sample Master Services Agreement
    ("https://www.sec.gov/Archives/edgar/data/1166036/000119312510034213/dex1024.htm", "fallback.html"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def download(url: str, filename: str) -> bool:
    """Download a single PDF with browser user-agent. Returns True if valid PDF."""
    out_path = DATA_DIR / filename
    print(f"Downloading {filename} from {url}...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        # Check it's actually a PDF
        if not data.startswith(b"%PDF"):
            print(f"  WARNING: {filename} is not a PDF (got {len(data)} bytes, starts with {data[:10]})")
            return False
        out_path.write_bytes(data)
        print(f"  OK: {filename} ({len(data):,} bytes)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


if __name__ == "__main__":
    success = 0
    for url, name in PDFS:
        if download(url, name):
            success += 1
    print(f"\n{success}/{len(PDFS)} PDFs downloaded successfully.")
