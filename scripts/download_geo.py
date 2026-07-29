import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import requests
from tqdm import tqdm
from biovision_omics.checksums import sha256_file


def geo_range(accession: str) -> str:
    match = re.fullmatch(r"GSE(\d+)", accession.upper())
    if not match:
        raise ValueError("Accession must look like GSE292268")
    digits = match.group(1)
    return "GSEnnn" if len(digits) <= 3 else f"GSE{digits[:-3]}nnn"


def archive_url(accession: str) -> str:
    accession = accession.upper()
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_range(accession)}/{accession}/suppl/{accession}_RAW.tar"


def download(url: str, destination: Path, retries: int = 4) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as response:
                response.raise_for_status()
                if existing and response.status_code == 200:
                    existing = 0
                    partial.unlink(missing_ok=True)
                remaining = int(response.headers.get("content-length", 0)) or None
                mode = "ab" if existing else "wb"
                with partial.open(mode) as handle, tqdm(
                    total=(existing + remaining) if remaining else None,
                    initial=existing,
                    unit="B", unit_scale=True, desc=destination.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))
                partial.replace(destination)
                return {
                    "url": url,
                    "destination": str(destination),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                }
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", default="GSE292268")
    parser.add_argument("--output-root", default="data/raw")
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    accession = args.accession.upper()
    url = archive_url(accession)
    destination = Path(args.output_root) / accession / f"{accession}_RAW.tar"
    manifest = Path(args.manifest_root) / f"{accession}_download.json"
    if args.dry_run:
        print(json.dumps({"accession": accession, "url": url, "destination": str(destination)}, indent=2))
        return
    metadata = download(url, destination)
    metadata["accession"] = accession
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved archive: {destination}")
    print(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    main()
