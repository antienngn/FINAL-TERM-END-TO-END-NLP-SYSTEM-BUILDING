"""PDF scraper cho VNU/UET RAG knowledge resource.

Tải PDF từ URL, extract text bằng pdfplumber, lưu JSONL.

Usage:
    # Crawl 1 URL (smoke test)
    python scrape_pdf.py --url https://vnu.edu.vn/.../some.pdf

    # Crawl theo seeds.yaml
    python scrape_pdf.py --seeds scrape/seeds.yaml \\
        --section pdf_documents --tier high
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import pdfplumber
import requests
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
    "RAG-Research-Bot/1.0 (academic; contact: student@vnu.edu.vn)"
)
TIMEOUT = 30  # PDF có thể chậm hơn HTML
PDF_SIZE_LIMIT_MB = 50
MIN_TEXT_CHARS = 200  # Dưới mức này coi là scanned PDF (chỉ ảnh, không text layer)


@dataclass
class PdfDocument:
    id: str
    url: str
    final_url: str
    title: str
    text: str
    source_type: str
    tag: str | None
    doc_id: str | None        # Số hiệu văn bản (e.g., "10/2009/TT-BGDĐT")
    fetched_at: str
    http_status: int
    content_length: int
    page_count: int
    pdf_size_bytes: int


def make_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def fetch_pdf(url: str, session: requests.Session) -> requests.Response:
    resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    return resp


def clean_pdf_text(text: str) -> str:
    """Vietnamese-safe normalization for PDF-extracted text."""
    text = unicodedata.normalize("NFC", text)
    # Remove hyphen-line-break artifacts (PDF wrap)
    text = re.sub(r"-\n", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int, int]:
    """Extract text từ PDF bytes, page-aware.

    Returns:
        (text, page_count, pages_with_text)
        — text có markers "=== Page N ===" giữa các trang.
        — pages_with_text giúp detect scanned PDFs (gần 0 nếu là scan).
    """
    text_parts = []
    pages_with_text = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            page_text = clean_pdf_text(page_text)
            if page_text.strip():
                text_parts.append(f"=== Page {i} ===\n{page_text}")
                pages_with_text += 1
    return "\n\n".join(text_parts), page_count, pages_with_text


def url_to_title(url: str) -> str:
    """Extract filename từ URL, dùng làm fallback title."""
    path = urlparse(url).path
    fname = path.split("/")[-1]
    title = unquote(fname)
    # Bỏ extension
    title = re.sub(r"\.[pP][dD][fF]$", "", title)
    title = title.replace("_", " ").replace("%20", " ").strip()
    return title


def scrape_pdf(
    url: str,
    session: requests.Session,
    tag: str | None = None,
    doc_id: str | None = None,
    name: str | None = None,
) -> PdfDocument | None:
    try:
        resp = fetch_pdf(url, session)
    except Exception as e:
        print(f"  ✗ FETCH FAIL: {url} — {type(e).__name__}: {e}", file=sys.stderr)
        return None

    # Verify content-type hoặc đuôi .pdf
    ctype = resp.headers.get("content-type", "").lower()
    if "pdf" not in ctype and not url.lower().endswith(".pdf"):
        print(f"  ⊘ SKIP (not PDF): {url} — {ctype}", file=sys.stderr)
        return None

    pdf_bytes = resp.content
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > PDF_SIZE_LIMIT_MB:
        print(f"  ⊘ SKIP (too large): {url} — {size_mb:.1f}MB", file=sys.stderr)
        return None

    try:
        text, page_count, pages_with_text = extract_pdf_text(pdf_bytes)
    except Exception as e:
        print(f"  ✗ PARSE FAIL: {url} — {type(e).__name__}: {e}", file=sys.stderr)
        return None

    # Detect scanned PDF (no text layer) — needs OCR which we don't have
    if len(text) < MIN_TEXT_CHARS:
        print(
            f"  ⊘ SCANNED (skip): {pages_with_text}/{page_count} pages have text, "
            f"{len(text)} chars total — PDF cần OCR (chưa hỗ trợ)",
            file=sys.stderr,
        )
        return None

    title = name or url_to_title(url)

    return PdfDocument(
        id=make_id(url),
        url=url,
        final_url=resp.url,
        title=unicodedata.normalize("NFC", title),
        text=text,
        source_type="pdf",
        tag=tag,
        doc_id=doc_id,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        http_status=resp.status_code,
        content_length=len(text),
        page_count=page_count,
        pdf_size_bytes=len(pdf_bytes),
    )


def write_jsonl(docs: list[PdfDocument], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load_seeds(yaml_path: Path, section: str, tier: str) -> list[dict]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    sec = cfg.get(section, {})
    if not sec:
        raise SystemExit(f"Section '{section}' không tồn tại trong {yaml_path}")
    items = sec.get(tier, [])
    out = []
    for it in items:
        if "url" not in it:
            continue
        out.append({
            "url": it["url"],
            "tag": it.get("tag"),
            "doc_id": it.get("doc_id"),
            "name": it.get("name"),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", nargs="+", help="Crawl 1 hoặc nhiều PDF URL")
    p.add_argument("--seeds", help="Path tới seeds.yaml")
    p.add_argument("--section", default="pdf_documents")
    p.add_argument("--tier", default="high",
                   help="Tier name trong seeds.yaml")
    p.add_argument("--out", default="data/raw/pdf_documents.jsonl")
    p.add_argument("--rate", type=float, default=2.0, help="Sec/request")
    args = p.parse_args()

    if not args.url and not args.seeds:
        p.error("Phải có --url hoặc --seeds")

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "vi,en;q=0.9",
        "Accept": "application/pdf, */*",
    })

    seeds = (
        [{"url": u, "tag": "demo", "doc_id": None, "name": None} for u in args.url]
        if args.url
        else load_seeds(Path(args.seeds), args.section, args.tier)
    )

    print(f"→ Downloading {len(seeds)} PDF ({args.rate}s/req)")
    docs: list[PdfDocument] = []
    for i, s in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] {s['url'][:100]}")
        d = scrape_pdf(s["url"], session, tag=s["tag"], doc_id=s["doc_id"], name=s["name"])
        if d:
            print(f"  ✓ {d.http_status} | {d.page_count} pages | "
                  f"{d.pdf_size_bytes/1024:.0f}KB | {d.content_length:,} chars | "
                  f"title: {d.title[:60]!r}")
            docs.append(d)
        if i < len(seeds):
            time.sleep(args.rate)

    out_path = Path(args.out)
    write_jsonl(docs, out_path)
    print(f"\n→ Wrote {len(docs)}/{len(seeds)} PDFs → {out_path}")
    print(f"  Total content: {sum(d.content_length for d in docs):,} chars")
    print(f"  Total PDF size: {sum(d.pdf_size_bytes for d in docs)/(1024*1024):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
