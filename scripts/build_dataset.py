#!/usr/bin/env python
"""Download the raw UFC data and build the pre-fight feature table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from mma_predictor.download import download
from mma_predictor.features import build_features, save_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    print("1/2  Fetching raw CSVs")
    download(force=args.force_download)

    print("\n2/2  Building pre-fight features (one chronological pass)")
    frame = build_features()
    save_features(frame)
    print(
        f"\n  {len(frame):,} modelled bouts, "
        f"{frame.date.min().date()} to {frame.date.max().date()}"
    )
    print(f"  win rate in slot A: {frame.label.mean():.4f}  (0.5 = no ordering bias)")
    print(f"  bouts with a closing line: {frame.has_market.mean():.1%}")


if __name__ == "__main__":
    main()
