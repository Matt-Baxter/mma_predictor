#!/usr/bin/env python
"""Fill the gaps in the closing-line data by scraping BestFightOdds.

About a fifth of bouts have no closing line here, which is a hole in the public
odds datasets rather than in reality -- essentially every UFC fight is priced.
BestFightOdds carries all of them back to 2008, and `ufcscraper` already knows
how to read it and tie each price to a UFCStats fight and fighter id.

This runs in two stages:

    python scripts/fetch_bestfightodds.py            # scrape, then merge
    python scripts/fetch_bestfightodds.py --merge-only

The scrape is slow and hits a third-party site, so it defaults to a one-second
delay between requests. Run it once, keep the output, and use --merge-only
afterwards. Nothing here touches the model: odds are only ever a benchmark, so a
wider sample improves the evaluation, not the predictions.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mma_predictor.config import RAW_DIR
from mma_predictor.parsing import normalize_name

SCRAPE_DIR = RAW_DIR.parent / "bestfightodds"
MERGED = RAW_DIR / "bestfightodds_odds.csv"


def scrape(delay: float) -> None:
    """Run ufcscraper's two stages: UFCStats first (for ids), then the odds."""
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    stages = [
        ("UFCStats fights and fighters", "ufcscraper_scrape_ufcstats_data"),
        ("BestFightOdds prices", "ufcscraper_scrape_bestfightodds_data"),
    ]
    for label, command in stages:
        print(f"\n=== {label} ===")
        result = subprocess.run(
            [command, "--data-folder", str(SCRAPE_DIR), "--delay", str(delay)],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"\n'{command}' failed (exit {result.returncode}).\n"
                "If this was a connection or 403 error, the network is blocking\n"
                "bestfightodds.com -- run this on a machine with direct access."
            )


def merge() -> None:
    """Reduce the scraper's output to (fight_id, fighter, closing line)."""
    odds_path = SCRAPE_DIR / "BestFightOdds_odds.csv"
    fighters_path = SCRAPE_DIR / "fighter_data.csv"
    for path in (odds_path, fighters_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run without --merge-only first.")

    odds = pd.read_csv(odds_path)
    fighters = pd.read_csv(fighters_path)

    expected = {"fight_id", "fighter_id", "closing_range_min", "closing_range_max"}
    missing = expected - set(odds.columns)
    if missing:
        raise SystemExit(f"{odds_path} is missing columns {sorted(missing)}.")

    names = (
        fighters.fighter_f_name.fillna("").astype(str)
        + " "
        + fighters.fighter_l_name.fillna("").astype(str)
    )
    key_by_id = dict(zip(fighters.fighter_id, names.map(normalize_name), strict=True))

    # BestFightOdds reports a closing range across books; the midpoint is the
    # single number comparable to the Kaggle file's closing line.
    closing = odds[["closing_range_min", "closing_range_max"]].mean(axis=1)
    merged = pd.DataFrame(
        {
            "fight_id": odds.fight_id.astype(str),
            "name_key": odds.fighter_id.map(key_by_id),
            "closing_odds": closing.round().astype("Int64"),
            "opening_odds": odds.get("opening"),
        }
    ).dropna(subset=["name_key", "closing_odds"])
    merged = merged[merged.name_key.astype(bool)]
    merged.to_csv(MERGED, index=False)

    fights = pd.read_csv(RAW_DIR / "ufc_fights.csv", usecols=["fight_url"])
    known = set(fights.fight_url.astype(str).str.rsplit("/", n=1).str[-1])
    matched = merged.fight_id.isin(known).mean()
    print(f"\nWrote {MERGED}")
    print(f"  {len(merged):,} priced fighter-sides across {merged.fight_id.nunique():,} bouts")
    print(f"  {matched:.1%} of them match a bout in ufc_fights.csv")
    print("\nRebuild the features to pick them up:  python scripts/build_dataset.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge-only", action="store_true",
                        help="skip the scrape and just rebuild the merged file")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between requests (default 1.0; be polite)")
    args = parser.parse_args()
    if not args.merge_only:
        scrape(args.delay)
    merge()


if __name__ == "__main__":
    main()
