"""Web scraper cho VNU/UET RAG knowledge resource.

Usage:
    # Crawl 1 URL (smoke test)
    python scrape_web.py --url https://vnu.edu.vn/

    # Crawl theo seeds.yaml
    python scrape_web.py --seeds scrape/seeds.yaml --section vnu_main --tier high
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
    "RAG-Research-Bot/1.0 (academic; contact: student@vnu.edu.vn)"
)
TIMEOUT = 15

# Always-strip: pure noise, never has useful content
STRIP_ALWAYS = ["script", "style", "noscript", "iframe", "svg"]

# Conditional-strip: only remove if a proper content container exists.
# Some SPA/landing pages put real content inside <header>/<nav>, so we must
# verify there's content elsewhere before stripping these.
STRIP_IF_CONTENT = [
    "nav", "header", "footer",
    ".navbox", ".infobox", "sup.reference",
    "#mw-navigation", "#footer", "#mw-head",
    ".navigation", ".breadcrumb", ".sidebar",
    ".social-share", ".comments", ".related-posts",
]


@dataclass
class Document:
    id: str
    url: str
    final_url: str
    title: str
    text: str
    source_type: str
    tag: str | None
    fetched_at: str
    http_status: int
    content_length: int


def make_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def fetch(url: str, session: requests.Session) -> requests.Response:
    resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    return resp


MIN_CONTENT_CHARS = 500  # threshold: when content container is "real"


def extract_text(html: str, url: str) -> tuple[str, str]:
    """Return (title, clean_text). Smart container selection + tiered stripping."""
    soup = BeautifulSoup(html, "lxml")

    # Always-strip noise (safe everywhere)
    for sel in STRIP_ALWAYS:
        for tag in soup.select(sel):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and not title:
        title = h1.get_text(strip=True)

    # Find best content container. Some sites (Next.js/React SSR) leave <main>
    # empty, putting real content in sibling <div>. Pick the one with most text.
    candidates = [
        soup.select_one(".mw-parser-output"),  # wikipedia preferred
        soup.find("article"),
        soup.find("main"),
        soup.body,
    ]
    container = max(
        (c for c in candidates if c),
        key=lambda c: len(c.get_text(" ", strip=True)),
        default=soup,
    )

    # Conditional strip: only if container has substantial content elsewhere.
    # On SPA landing pages, header/nav may BE the content — don't strip blindly.
    container_text_len = len(container.get_text(" ", strip=True))
    if container_text_len >= MIN_CONTENT_CHARS:
        for sel in STRIP_IF_CONTENT:
            for tag in container.select(sel):
                tag.decompose()

    blocks = []
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        txt = el.get_text(" ", strip=True)
        if txt and len(txt) > 1:
            blocks.append(txt)

    # Fallback: if find_all yields nothing (rare layout), use raw container text
    if not blocks:
        raw = container.get_text("\n", strip=True)
        if raw:
            blocks = [line for line in raw.split("\n") if len(line.strip()) > 1]

    text = "\n".join(blocks)
    return title, text


def clean_text(text: str) -> str:
    """Vietnamese-safe text normalization."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[​-‏﻿]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "wikipedia.org" in host:
        return "wikipedia"
    if "uet.vnu.edu.vn" in host:
        return "uet"
    if "vnu.edu.vn" in host:
        return "vnu_main"
    return "other"


def scrape_url(url: str, session: requests.Session, tag: str | None = None) -> Document | None:
    try:
        resp = fetch(url, session)
    except Exception as e:
        print(f"  ✗ FETCH FAIL: {url} — {type(e).__name__}: {e}", file=sys.stderr)
        return None

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype.lower():
        print(f"  ⊘ SKIP (not HTML): {url} — {ctype}", file=sys.stderr)
        return None

    title, raw_text = extract_text(resp.text, resp.url)
    text = clean_text(raw_text)

    return Document(
        id=make_id(url),
        url=url,
        final_url=resp.url,
        title=unicodedata.normalize("NFC", title),
        text=text,
        source_type=detect_source_type(resp.url),
        tag=tag,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        http_status=resp.status_code,
        content_length=len(text),
    )


def write_jsonl(docs: list[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load_seeds(yaml_path: Path, section: str, tier: str) -> list[dict]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    sec = cfg.get(section, {})
    if not sec:
        raise SystemExit(f"Section '{section}' không tồn tại trong {yaml_path}")
    base = sec.get("base_url", "")
    items = sec.get(tier, [])
    out = []
    for it in items:
        if "url" in it:
            url = it["url"]
        elif "path" in it:
            url = urljoin(base + "/", it["path"].lstrip("/"))
        else:
            continue
        out.append({"url": url, "tag": it.get("tag")})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", nargs="+", help="Crawl 1 hoặc nhiều URL (smoke test)")
    p.add_argument("--seeds", help="Path tới seeds.yaml")
    p.add_argument("--section", default="vnu_main")
    p.add_argument("--tier", default="high",
                   help="Tier name trong seeds.yaml (high/med/subdomains/...)")
    p.add_argument("--out", default="data/raw/web.jsonl")
    p.add_argument("--rate", type=float, default=1.0, help="Sec/request")
    args = p.parse_args()

    if not args.url and not args.seeds:
        p.error("Phải có --url hoặc --seeds")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.9"})

    seeds = (
        [{"url": u, "tag": "demo"} for u in args.url]
        if args.url
        else load_seeds(Path(args.seeds), args.section, args.tier)
    )

    print(f"→ Crawling {len(seeds)} URL ({args.rate}s/req)")
    docs: list[Document] = []
    for i, s in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] {s['url']}")
        d = scrape_url(s["url"], session, tag=s.get("tag"))
        if d:
            print(f"  ✓ {d.http_status} | {len(d.text):,} chars | title: {d.title[:70]!r}")
            docs.append(d)
        if i < len(seeds):
            time.sleep(args.rate)

    out_path = Path(args.out)
    write_jsonl(docs, out_path)
    print(f"\n→ Wrote {len(docs)}/{len(seeds)} docs → {out_path}")
    print(f"  Total content: {sum(d.content_length for d in docs):,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
