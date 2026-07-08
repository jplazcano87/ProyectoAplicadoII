"""Scrapes the RAN (bancos) index from cmfchile.cl and downloads chapter PDFs.

Pipeline:
  parse_index()      -> [(chapter_code, title, chapter_url), ...]
  resolve_pdf_url()  -> direct URL to the chapter PDF (runtime discovery)
  download_pdf()     -> cached download to data/raw/<code>.pdf + sha256

A manifest.json is written to data/raw/ for traceability/reproducibility.

Network access is only needed here; the rest of the corpus pipeline (parse,
chunk) works offline from data/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

# Server-rendered index of the "Recopilación Actualizada de Normas" for banks.
# The old SBIF URL (w3-propertyvalue-29580.html) redirects here.
INDEX_URL = (
    "https://www.cmfchile.cl/portal/normativa/624/"
    "w4-propertyvalue-49163.html?indice=3.1.2&LNAN=1"
)

USER_AGENT = "NormaRAG/0.1 (academic research; contact: jp.lazcano87@gmail.com)"

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"

# "Capítulo 1-13", "Capítulo 20-9", etc.
CHAPTER_RE = re.compile(r"Cap[ií]tulo\s+(\d+-\d+)", re.IGNORECASE)

# Accessibility note the site appends to link text, e.g.
# "Accionistas. Disposiciones varias., se abre en nueva ventana".
NEW_WINDOW_RE = re.compile(r",?\s*se abre en (una )?nueva ventana", re.IGNORECASE)

REQUEST_TIMEOUT = 30.0
RATE_LIMIT_SECONDS = 1.5
MAX_RETRIES = 3


@dataclass
class ChapterRef:
    chapter_code: str
    title: str
    chapter_url: str


@dataclass
class ManifestEntry:
    chapter_code: str
    title: str
    chapter_url: str
    pdf_url: str
    path: str
    sha256: str
    bytes: int


def _clean_title(text: str) -> str:
    """Strip the 'Capítulo N-M' prefix, accessibility note, and wrapping punctuation."""
    title = CHAPTER_RE.sub("", text, count=1)
    title = NEW_WINDOW_RE.sub("", title)
    # Links are wrapped like "( <title>)"; strip surrounding parens/punct/space.
    return title.strip().strip("()").strip(" .-–—:")


def _is_pdf_url(url: str) -> bool:
    """True if the URL path ends in .pdf, ignoring any query string (?ts=...)."""
    return urlparse(url).path.lower().endswith(".pdf")


def _natural_key(code: str) -> tuple[int, int]:
    """Sort '1-1', '1-3', '2-1' naturally by (major, minor)."""
    major, _, minor = code.partition("-")
    return (int(major), int(minor))


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with retries and rate limiting."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            time.sleep(RATE_LIMIT_SECONDS)
            return resp
        except (httpx.HTTPError, httpx.TransportError) as exc:  # pragma: no cover - network
            last_exc = exc
            wait = RATE_LIMIT_SECONDS * attempt * 2
            print(f"  ! request failed ({exc!r}); retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {MAX_RETRIES} retries: {url}") from last_exc


def parse_index(html: str, base_url: str = INDEX_URL) -> list[ChapterRef]:
    """Extract (chapter_code, title, chapter_url) from the index HTML.

    The index lists each chapter as a link whose text is like
    "Capítulo 1-13 Clasificación de gestión y solvencia". Entries are
    duplicated; we deduplicate by chapter_code keeping the longest title.
    """
    soup = BeautifulSoup(html, "html.parser")
    best: dict[str, ChapterRef] = {}

    for link in soup.find_all("a", href=True):
        text = " ".join(link.get_text(" ", strip=True).split())
        match = CHAPTER_RE.search(text)
        if not match:
            continue
        code = match.group(1)
        title = _clean_title(text)
        url = urljoin(base_url, link["href"])

        current = best.get(code)
        if current is None or len(title) > len(current.title):
            best[code] = ChapterRef(chapter_code=code, title=title, chapter_url=url)

    return sorted(best.values(), key=lambda c: _natural_key(c.chapter_code))


def resolve_pdf_url(client: httpx.Client, chapter: ChapterRef) -> str | None:
    """Discover the direct PDF URL for a chapter at runtime.

    If the chapter link already points to a .pdf, use it. Otherwise open the
    chapter page and find the first .pdf link, preferring the historical
    "_doc_pdf" pattern.
    """
    if _is_pdf_url(chapter.chapter_url):
        return chapter.chapter_url

    resp = _get(client, chapter.chapter_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    pdf_links = [
        urljoin(chapter.chapter_url, a["href"])
        for a in soup.find_all("a", href=True)
        if ".pdf" in a["href"].lower()
    ]
    if not pdf_links:
        return None

    for link in pdf_links:
        if "_doc_pdf" in link.lower():
            return link
    return pdf_links[0]


def _sha256_and_size(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), len(data)


def download_pdf(client: httpx.Client, chapter: ChapterRef, pdf_url: str) -> ManifestEntry:
    """Download a chapter PDF to data/raw/<code>.pdf, caching if it exists."""
    path = RAW_DIR / f"{chapter.chapter_code}.pdf"

    if path.exists():
        data = path.read_bytes()
        sha, size = _sha256_and_size(data)
        print(f"  = cached {path} ({size} bytes)")
    else:
        resp = _get(client, pdf_url)
        data = resp.content
        path.write_bytes(data)
        sha, size = _sha256_and_size(data)
        print(f"  + downloaded {path} ({size} bytes)")

    return ManifestEntry(
        chapter_code=chapter.chapter_code,
        title=chapter.title,
        chapter_url=chapter.chapter_url,
        pdf_url=pdf_url,
        path=str(path),
        sha256=sha,
        bytes=size,
    )


def download_corpus(limit: int | None = None) -> list[ManifestEntry]:
    """Full download flow: parse index, resolve PDFs, download, write manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        print(f"Fetching index: {INDEX_URL}")
        index_resp = _get(client, INDEX_URL)
        chapters = parse_index(index_resp.text)
        print(f"Found {len(chapters)} unique chapters.")

        if limit is not None:
            chapters = chapters[:limit]
            print(f"Limiting to first {len(chapters)} chapters.")

        entries: list[ManifestEntry] = []
        for chapter in chapters:
            print(f"\n[{chapter.chapter_code}] {chapter.title}")
            print(f"  chapter_url: {chapter.chapter_url}")
            pdf_url = resolve_pdf_url(client, chapter)
            if pdf_url is None:
                print("  ! no PDF link found; skipping")
                continue
            print(f"  pdf_url: {pdf_url}")
            entries.append(download_pdf(client, chapter, pdf_url))

    MANIFEST_PATH.write_text(
        json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote manifest with {len(entries)} entries -> {MANIFEST_PATH}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the RAN (bancos) corpus.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N chapters (for validation).",
    )
    args = parser.parse_args()
    download_corpus(limit=args.limit)


if __name__ == "__main__":
    main()
