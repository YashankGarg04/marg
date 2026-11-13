"""
Scraper for https://www.myscheme.gov.in  (all languages)

The website is a JavaScript app, so instead of parsing HTML this script calls the
same JSON API that the site calls in your browser.

IMPORTANT: the endpoints, API key and language codes below are what the site used
when I last knew it. They can change. If you get 401/403/404 errors, open the site,
press F12 -> Network tab -> Fetch/XHR, reload the search page, and copy the current
request URL + the `x-api-key` header into the CONFIG section.

Output (per language):  data/<lang>/schemes_list.jsonl   (search results)
                        data/<lang>/schemes_detail.jsonl (full scheme details)
The script is resumable: re-run it and it skips slugs it already saved.

Usage:
    pip install requests
    python myscheme_scraper.py                 # all languages, list + details
    python myscheme_scraper.py --langs en hi   # only some languages
    python myscheme_scraper.py --list-only     # skip detail pages
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------- CONFIG ---------------------------------------
SEARCH_URL = "https://www.myscheme.gov.in/api/apisetu/search/schemes"
DETAIL_URL = "https://api.myscheme.gov.in/schemes/v6/public/schemes"
# Public key that the website's own JS sends. Re-copy from DevTools if it stops working.
API_KEY = "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc"

# Verify these against the language dropdown / the `lang=` param in DevTools.
LANGUAGES = [
    "en", "hi", "as", "bn", "gu", "kn", "ml", "mr",
    "or", "pa", "ta", "te", "ur", "ne", "sd",
]

PAGE_SIZE = 10          # results per search request (10 is what the site itself uses)
DELAY = (0.5, 1.2)      # random polite delay (seconds) between requests
OUT_DIR = Path("data")
# -----------------------------------------------------------------------------

HEADERS = {
    "x-api-key": API_KEY,
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.myscheme.gov.in",
    "Referer": "https://www.myscheme.gov.in/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    # Visit the home page first so the session picks up any cookies the site sets.
    try:
        s.get("https://www.myscheme.gov.in/", timeout=30)
        s.get("https://www.myscheme.gov.in/api/auth/digilocker/session", timeout=30)
    except Exception as e:
        print("Warning: warm-up request failed:", e)
    return s


def pause():
    time.sleep(random.uniform(*DELAY))


def fetch_list_page(s, lang, offset):
    params = {
        "lang": lang,
        "q": "[]",          # no filters
        "keyword": "",
        "sort": "",
        "from": offset,
        "size": PAGE_SIZE,
    }
    r = s.get(SEARCH_URL, params=params, timeout=30)
    if not r.ok:
        print("Search failed:", r.status_code, r.text[:300])
    r.raise_for_status()
    body = r.json()
    data = body.get("data", body)
    items = data.get("hits", {}).get("items", [])
    total = data.get("summary", {}).get("total", 0)
    return items, total


def load_done_slugs(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("slug"))
                except json.JSONDecodeError:
                    pass
    return done


def scrape_list(s, lang, out_path: Path) -> list:
    """Download every scheme in the search listing for one language."""
    done = load_done_slugs(out_path)
    all_slugs = []
    offset, total = 0, None
    with out_path.open("a", encoding="utf-8") as f:
        while total is None or offset < total:
            items, total = fetch_list_page(s, lang, offset)
            if not items:
                break
            for it in items:
                fields = it.get("fields", {})
                slug = fields.get("slug")
                all_slugs.append(slug)
                if slug in done:
                    continue
                record = {"id": it.get("id"), "slug": slug, **fields}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            offset += PAGE_SIZE
            print(f"[{lang}] list {min(offset, total)}/{total}")
            pause()
    return all_slugs


def scrape_details(s, lang, slugs, out_path: Path):
    """Download the full detail (eligibility, benefits, documents, ...) per scheme."""
    done = load_done_slugs(out_path)
    todo = [x for x in slugs if x and x not in done]
    print(f"[{lang}] details: {len(todo)} to fetch ({len(done)} already saved)")
    with out_path.open("a", encoding="utf-8") as f:
        for i, slug in enumerate(todo, 1):
            try:
                r = s.get(DETAIL_URL, params={"slug": slug, "lang": lang}, timeout=30)
                r.raise_for_status()
                body = r.json()
                payload = body.get("data") if isinstance(body, dict) else None
                if not isinstance(payload, dict) or not payload:
                    raise ValueError("empty or missing 'data' in response")
            except Exception as e:  # keep going on individual failures
                print(f"[{lang}] FAILED {slug}: {e}")
                with (out_path.parent / "failed_slugs.txt").open("a", encoding="utf-8") as ff:
                    ff.write(f"{slug}\n")
                continue
            f.write(json.dumps({"slug": slug, "lang": lang, **payload},
                               ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"[{lang}] details {i}/{len(todo)}")
            pause()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGUAGES)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    s = make_session()
    for lang in args.langs:
        lang_dir = OUT_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {lang} =====")
        try:
            slugs = scrape_list(s, lang, lang_dir / "schemes_list.jsonl")
            if not args.list_only:
                scrape_details(s, lang, slugs, lang_dir / "schemes_detail.jsonl")
        except requests.HTTPError as e:
            print(f"[{lang}] HTTP error, skipping language: {e}")

    print("\nDone. Output in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()