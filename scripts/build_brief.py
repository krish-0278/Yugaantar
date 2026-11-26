"""Build brief/brief.pdf from brief/brief.html using installed headless Edge and verify page count == 2."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
BRIEF_HTML = ROOT / "brief" / "brief.html"
BRIEF_PDF = ROOT / "brief" / "brief.pdf"


def find_browser() -> str:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for path in candidates:
        if Path(path).is_file():
            return path
    found = shutil.which("msedge") or shutil.which("chrome")
    if found:
        return found
    raise FileNotFoundError("Neither msedge nor chrome was found on this machine.")


def count_pdf_pages(pdf_path: Path) -> int:
    data = pdf_path.read_bytes()
    counts = [int(m.group(1)) for m in re.finditer(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", data, flags=re.S)]
    if counts:
        return max(counts)
    pages = re.findall(rb"/Type\s*/Page\b(?!s)", data)
    return len(pages)


def build_pdf() -> int:
    if not BRIEF_HTML.is_file():
        raise FileNotFoundError(f"Missing {BRIEF_HTML}")

    browser = find_browser()
    BRIEF_PDF.unlink(missing_ok=True)
    prof_dir = Path(tempfile.gettempdir()) / "edge_brief_pdf"
    shutil.rmtree(prof_dir, ignore_errors=True)

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={prof_dir}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={BRIEF_PDF.resolve()}",
        BRIEF_HTML.resolve().as_uri(),
    ]
    try:
        res = subprocess.run(cmd, timeout=30, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"PDF print failed ({res.returncode}): {res.stderr}")
    finally:
        shutil.rmtree(prof_dir, ignore_errors=True)

    if not BRIEF_PDF.is_file():
        raise RuntimeError("brief.pdf was not created")

    n_pages = count_pdf_pages(BRIEF_PDF)
    print(f"Built {BRIEF_PDF} ({BRIEF_PDF.stat().st_size} bytes), page_count={n_pages}")
    if n_pages != 2:
        raise RuntimeError(f"Expected exactly 2 pages in brief.pdf, got {n_pages}")
    return n_pages


if __name__ == "__main__":
    build_pdf()
