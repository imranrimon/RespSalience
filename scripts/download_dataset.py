import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Set
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

BASE_URL = "https://physionet.org/files/mimic4wdb/0.1.0/"
OUTPUT_ROOT = Path("data/mimic4wdb")
HREF_RE = re.compile(r'href=[\"\']([^\"\']+)[\"\']', re.IGNORECASE)


def _fetch_text(url: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _list_links(dir_url: str) -> List[str]:
    html = _fetch_text(dir_url)
    links = []
    for href in HREF_RE.findall(html):
        if href in ("../", "./"):
            continue
        if href.startswith("#"):
            continue
        links.append(href)
    return links


def _crawl_files(base_url: str) -> List[str]:
    files: List[str] = []
    queue = deque([base_url])
    seen: Set[str] = set()

    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)

        try:
            links = _list_links(cur)
        except Exception as exc:
            print(f"[WARN] Failed to list {cur}: {exc}")
            continue

        for href in links:
            full = urljoin(cur, href)
            if not full.startswith(base_url):
                continue
            if href.endswith("/"):
                queue.append(full)
            else:
                files.append(full)

        print(f"[CRAWL] dirs={len(seen)} files={len(files)} current={cur}")

    return sorted(set(files))


def _remote_size(url: str) -> int:
    req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else -1


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    remote_size = -1
    try:
        remote_size = _remote_size(url)
    except Exception:
        pass

    if out_path.exists() and remote_size > 0 and out_path.stat().st_size == remote_size:
        return

    existing = out_path.stat().st_size if out_path.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if existing > 0 and remote_size > existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=120) as resp, out_path.open(mode) as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Crawling: {BASE_URL}")
    files = _crawl_files(BASE_URL)
    print(f"[INFO] Total files discovered: {len(files)}")

    if not files:
        print("[ERROR] No files discovered. Check your PhysioNet access permissions.")
        return 2

    ok = 0
    failed = 0
    start = time.time()

    prefix = urlparse(BASE_URL).path
    for idx, url in enumerate(files, start=1):
        rel = urlparse(url).path
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
        out_path = OUTPUT_ROOT / rel

        try:
            _download_file(url, out_path)
            ok += 1
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            failed += 1
            print(f"[WARN] {idx}/{len(files)} failed: {url} -> {exc}")

        if idx % 25 == 0 or idx == len(files):
            elapsed = time.time() - start
            print(f"[PROGRESS] {idx}/{len(files)} ok={ok} failed={failed} elapsed={elapsed:.1f}s")

    print(f"[DONE] ok={ok} failed={failed} output={OUTPUT_ROOT}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
