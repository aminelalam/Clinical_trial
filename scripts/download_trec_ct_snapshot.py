"""Download the official TREC 2021 Clinical Trials snapshot zips.

The TREC CT 2021/2022 benchmark corpus is the 2021-04-27 ClinicalTrials.gov
snapshot distributed in five zip parts by trec-cds.org. This script downloads
and optionally extracts those parts without applying any study filters.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

URLS = [
    "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part1.zip",
    "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part2.zip",
    "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part3.zip",
    "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part4.zip",
    "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part5.zip",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}", file=sys.stderr)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "trial-matcher/1.0"}) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def _extract(zip_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"already extracted: {zip_path.name}", file=sys.stderr)
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download-dir", type=Path, default=Path("data/trec_ct/downloads"))
    p.add_argument("--output-dir", type=Path, default=Path("data/trec_ct/clinical_trials_2021_04_27"))
    p.add_argument("--no-extract", action="store_true")
    args = p.parse_args()

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        dest = args.download_dir / url.rsplit("/", 1)[-1]
        print(f"download: {url}", file=sys.stderr)
        _download(url, dest)
        if not args.no_extract:
            print(f"extract: {dest}", file=sys.stderr)
            _extract(dest, args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

