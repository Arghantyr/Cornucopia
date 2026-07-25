"""Pixabay Continuous AI Dataset Downloader (v3) — Streamlined, atomic, zero-waste harvester."""

# IMPORTS
import argparse
import hashlib
import json
import logging
import os
import pathlib
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional, Tuple

import dotenv
import requests
from PIL import ExifTags, Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

# CONSTANTS
PIXABAY_API_URL: str = "https://pixabay.com/api/"
CACHE_TTL_SECONDS: int = 86400
DEFAULT_CACHE_DB: str = "pixabay_cache.sqlite"
DEFAULT_OUTPUT_DIR: str = "downloaded_images"
DEFAULT_PER_PAGE: int = 200
DEFAULT_TIMEOUT_SECONDS: int = 30
API_POLL_DELAY_SECONDS: float = 0.6

RESOLUTIONS: Tuple[str, ...] = ("imageURL", "fullHDURL", "largeImageURL", "webformatURL", "previewURL")
TAG_MAP: Dict[str, str] = {"imageURL": "original", "fullHDURL": "1920px", "largeImageURL": "1280px", "webformatURL": "640px", "previewURL": "150px"}
CATEGORIES: Tuple[str, ...] = ("nature", "science", "computer", "education", "people", "places", "animals", "industry", "food", "sports", "transportation", "travel", "buildings", "business", "music", "fashion", "feelings", "health", "religion", "backgrounds")


# CLASSES
class DatasetStore:
    """SQLite manager for 24h query caching and SHA-256 image cataloging."""

    def __init__(self, db_path: str = DEFAULT_CACHE_DB) -> None:
        self.db_path = db_path
        self._sql("CREATE TABLE IF NOT EXISTS api_cache (query_hash TEXT PRIMARY KEY, response_json TEXT, timestamp REAL)")
        self._sql("CREATE TABLE IF NOT EXISTS downloaded_images (image_id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, filename TEXT NOT NULL, download_timestamp_utc TEXT NOT NULL, license TEXT NOT NULL, metadata_json TEXT NOT NULL)")

    def _sql(self, query: str, params: tuple = (), fetch: bool = False) -> Any:
        """Centralized SQL dispatcher with guaranteed connection closure."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall() if fetch else None
        finally:
            conn.close()

    @staticmethod
    def _hash_params(params: Dict[str, Any]) -> str:
        """Deterministic MD5 hash for query parameter dictionaries."""
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def get_cached_query(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieves cached response if younger than 24 hours, purging expired entries."""
        query_hash = self._hash_params(params)
        rows = self._sql("SELECT response_json, timestamp FROM api_cache WHERE query_hash = ?", (query_hash,), fetch=True)
        if not rows:
            return None
        if time.time() - rows[0][1] < CACHE_TTL_SECONDS:
            return json.loads(rows[0][0])
        self._sql("DELETE FROM api_cache WHERE query_hash = ?", (query_hash,))
        return None

    def set_cached_query(self, params: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Stores API response payload in cache with current timestamp."""
        self._sql("INSERT OR REPLACE INTO api_cache VALUES (?, ?, ?)", (self._hash_params(params), json.dumps(data), time.time()))

    def has_image(self, image_id: int) -> bool:
        """Fast indexed check whether an image has already been ingested."""
        return len(self._sql("SELECT 1 FROM downloaded_images WHERE image_id = ?", (image_id,), fetch=True)) > 0

    def record_download(self, image_id: int, sha256: str, filename: str, timestamp_utc: str, license_str: str, metadata: Dict[str, Any]) -> None:
        """Records downloaded image metadata and SHA-256 checksum into catalog."""
        self._sql("INSERT OR REPLACE INTO downloaded_images VALUES (?, ?, ?, ?, ?, ?)", (image_id, sha256, filename, timestamp_utc, license_str, json.dumps(metadata)))


class ResilientApiClient:
    """HTTP client with automatic 429/5xx backoff retries and 24h query caching."""

    def __init__(self, api_key: str, store: DatasetStore) -> None:
        self.api_key, self.store = api_key, store
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504]))
        self.session.mount("https://", adapter)

    def search_images(self, **kwargs: Any) -> Dict[str, Any]:
        """Queries Pixabay API, utilizing 24-hour cache when available."""
        cache_params = {k: v for k, v in kwargs.items() if v is not None}
        cached = self.store.get_cached_query(cache_params)
        if cached:
            return cached

        time.sleep(API_POLL_DELAY_SECONDS)
        try:
            resp = self.session.get(PIXABAY_API_URL, params={"key": self.api_key, **cache_params}, timeout=DEFAULT_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            self.store.set_cached_query(cache_params, data)
            return data
        except requests.RequestException as err:
            logging.error("API request failed: %s", err)
            return {"hits": [], "totalHits": 0}


class ContinuousIngestor:
    """Stream downloader with single-pass SHA-256, EXIF extraction, and atomic sidecar generation."""

    def __init__(self, output_dir: str, store: DatasetStore) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = store

    @staticmethod
    def extract_exif(image_path: pathlib.Path) -> Dict[str, Any]:
        """Extracts human-readable EXIF tags safely using public Pillow API."""
        try:
            with Image.open(image_path) as img:
                exif = img.getexif() or {}
                return {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in exif.items() if not isinstance(v, bytes)}
        except Exception:
            return {}

    def process_hit(self, api_client: ResilientApiClient, hit: Dict[str, Any]) -> bool:
        """Downloads image, computes SHA-256, extracts EXIF, writes sidecar — all-or-nothing atomic."""
        image_id = hit["id"]
        if self.store.has_image(image_id):
            return False

        res_key = next((k for k in RESOLUTIONS if hit.get(k)), None)
        if not res_key:
            return False

        image_url, res_tag = hit[res_key], TAG_MAP.get(res_key, "standard")
        ext = pathlib.Path(image_url.split("?")[0]).suffix or ".jpg"
        base_name = f"pixabay_{image_id}_{res_tag}"
        final_img = self.output_dir / f"{base_name}{ext}"
        tmp_img = self.output_dir / f"{base_name}.tmp"
        sidecar_json = self.output_dir / f"{base_name}.json"

        if final_img.exists() and sidecar_json.exists():
            return False

        # Phase 1: Stream download to .tmp with single-pass SHA-256
        sha256_hasher = hashlib.sha256()
        try:
            resp = api_client.session.get(image_url, stream=True, timeout=DEFAULT_TIMEOUT_SECONDS)
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))

            with open(tmp_img, "wb") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc=base_name, leave=False, disable=logging.getLogger().level > logging.INFO) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        sha256_hasher.update(chunk)
                        pbar.update(len(chunk))
        except Exception as exc:
            logging.error("Download failed for image ID %s: %s", image_id, exc)
            if tmp_img.exists():
                tmp_img.unlink()
            return False

        # Phase 2: Atomic commit — image rename + sidecar + SQLite (all-or-nothing)
        sha256_hex = sha256_hasher.hexdigest()
        try:
            tmp_img.rename(final_img)

            timestamp_utc = datetime.now(timezone.utc).isoformat()
            license_str = hit.get("license") or "Pixabay Content License"
            metadata_payload = {
                "sha256": sha256_hex,
                "timestamp": timestamp_utc,
                "license": license_str,
                "metadata": {"exif": self.extract_exif(final_img), "pixabay": hit},
            }

            with open(sidecar_json, "w", encoding="utf-8") as jf:
                json.dump(metadata_payload, jf, indent=2, ensure_ascii=False)

            self.store.record_download(image_id, sha256_hex, final_img.name, timestamp_utc, license_str, metadata_payload)
            logging.info("Saved: %s (SHA-256: %s...)", final_img.name, sha256_hex[:8])
            return True

        except Exception as exc:
            logging.error("Post-download processing failed for image ID %s: %s", image_id, exc)
            for orphan in (final_img, sidecar_json):
                if orphan.exists():
                    orphan.unlink()
            return False


# FUNCTIONS
def setup_logging(verbose: bool) -> None:
    """Configures structured console logging."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_api_key(cli_key: Optional[str]) -> str:
    """Resolves Pixabay API key from CLI argument, environment variable, or .env file."""
    dotenv.load_dotenv()
    key = cli_key or os.getenv("PIXABAY_API_KEY")
    if not key:
        logging.critical("Pixabay API key missing! Provide --key or set PIXABAY_API_KEY env var.")
        sys.exit(1)
    return key


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Pixabay Continuous Downloader v3 — Atomic AI dataset harvester.")
    parser.add_argument("--key", type=str, help="Pixabay API Key (or set PIXABAY_API_KEY env var)")
    parser.add_argument("-q", "--query", type=str, default="", help="Search term. Omit for multi-category rotation.")
    parser.add_argument("--category", type=str, choices=CATEGORIES, help="Filter by specific category")
    parser.add_argument("--image-type", type=str, choices=("all", "photo", "illustration", "vector"), default="photo")
    parser.add_argument("--orientation", type=str, choices=("all", "horizontal", "vertical"), default="all")
    parser.add_argument("--editors-choice", action="store_true", help="Editor's Choice awards only")
    parser.add_argument("--safesearch", action="store_true", default=True, help="Enable SafeSearch filter")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages per query (1-3)")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output dataset directory")
    parser.add_argument("--db-path", type=str, default=DEFAULT_CACHE_DB, help="SQLite database catalog path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def iter_search_pages(args: argparse.Namespace) -> Generator[Tuple[Dict[str, Any], int], None, None]:
    """Generates (search_params, page_number) tuples for category/query rotation with pagination."""
    terms = [(args.query, args.category)] if (args.query or args.category) else [("", c) for c in CATEGORIES]
    for query_str, cat in terms:
        for page in range(1, args.max_pages + 1):
            params = {
                "q": query_str,
                "category": cat,
                "image_type": args.image_type,
                "orientation": args.orientation,
                "editors_choice": "true" if args.editors_choice else "false",
                "safesearch": "true" if args.safesearch else "false",
                "per_page": DEFAULT_PER_PAGE,
                "page": page,
            }
            yield params, page


# MAIN
def main() -> None:
    """Orchestrates continuous dataset harvesting with smart pagination and atomic file operations."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logging.info("Starting Pixabay Continuous AI Dataset Downloader v3...")

    store = DatasetStore(db_path=args.db_path)
    api_client = ResilientApiClient(api_key=load_api_key(args.key), store=store)
    ingestor = ContinuousIngestor(output_dir=args.output_dir, store=store)

    total_downloaded = 0
    current_category = None
    try:
        for search_params, page in iter_search_pages(args):
            cat = search_params.get("category")
            if cat != current_category:
                current_category = cat
                logging.info("Processing category: '%s'", cat or "all")

            resp = api_client.search_images(**search_params)
            hits = resp.get("hits", [])
            total_available = resp.get("totalHits", 0)

            if not hits:
                break

            for hit in hits:
                if ingestor.process_hit(api_client, hit):
                    total_downloaded += 1

            # Stop paginating when all accessible results have been covered
            if page * DEFAULT_PER_PAGE >= total_available:
                break

    except KeyboardInterrupt:
        logging.info("Harvesting interrupted. Saved state preserved in SQLite.")

    logging.info("Harvesting complete. Ingested %d new images into '%s'.", total_downloaded, args.output_dir)


# EXECUTION
if __name__ == "__main__":
    main()

