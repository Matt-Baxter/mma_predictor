"""Fetch the five raw TidyTuesday UFC CSVs into ``data/raw``."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .config import (
    DATASETS,
    RAW_DIR,
    TIDYTUESDAY_BASE,
    UFCSTATS_BASE,
    UFCSTATS_DATASETS,
)


def download(force: bool = False, dest: Path = RAW_DIR) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sources = [(name, TIDYTUESDAY_BASE) for name in DATASETS]
    sources += [(name, UFCSTATS_BASE) for name in UFCSTATS_DATASETS]
    for name, base in sources:
        target = dest / f"{name}.csv"
        if target.exists() and not force:
            print(f"  [skip] {target.name} already present ({target.stat().st_size:,} bytes)")
            written.append(target)
            continue
        url = f"{base}/{name}.csv"
        print(f"  [get ] {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        target.write_bytes(payload)
        print(f"  [ok  ] {target.name} ({len(payload):,} bytes)")
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    print(f"Downloading TidyTuesday 2026-07-07 UFC data into {RAW_DIR}")
    download(force=args.force)


if __name__ == "__main__":
    main()
