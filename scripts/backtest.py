#!/usr/bin/env python
"""Replay real fights the model never trained on, and score its calls.

Every bout shown here comes from the held-out test split, and each is priced
from the fighters' careers *as they stood the day before the fight*. The actual
result and the closing line are printed next to the model's number, so you can
see where it was right, where it was wrong, and where it disagreed with the
market.

    python scripts/backtest.py --fighter "Sean O'Malley"
    python scripts/backtest.py --event "UFC 300"
    python scripts/backtest.py --upsets          # correctly called underdogs
    python scripts/backtest.py --disagreements   # biggest gaps vs the market
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mma_predictor.config import ARTIFACT_DIR, SplitConfig
from mma_predictor.dataset import prepare
from mma_predictor.features import load_features
from mma_predictor.parsing import normalize_name
from mma_predictor.train import ensemble_probabilities, load_checkpoint, to_tensors


def build_table() -> pd.DataFrame:
    models, transform, payload = load_checkpoint(ARTIFACT_DIR / "model.pt")
    frame = load_features()
    _, _, test_arrays, _, folds = prepare(frame, SplitConfig(**payload["config"]["splits"]))
    test = folds["test"].copy()
    test["model_prob_a"] = ensemble_probabilities(models, to_tensors(test_arrays))

    # Orient every row so the model's pick is "the pick", regardless of slot.
    picked_a = test.model_prob_a >= 0.5
    test["pick"] = np.where(picked_a, test.a_name, test.b_name)
    test["opponent"] = np.where(picked_a, test.b_name, test.a_name)
    test["pick_prob"] = np.where(picked_a, test.model_prob_a, 1 - test.model_prob_a)
    test["winner"] = np.where(test.label == 1, test.a_name, test.b_name)
    test["model_right"] = test.pick == test.winner
    test["market_prob_pick"] = np.where(
        picked_a, test.market_prob_a, 1 - test.market_prob_a
    )
    test["market_pick"] = np.where(test.market_prob_a >= 0.5, test.a_name, test.b_name)
    test["market_right"] = np.where(
        test.market_prob_a.isna(), np.nan, (test.market_pick == test.winner).astype(float)
    )
    test["disagreement"] = (test.pick_prob - test.market_prob_pick).abs()
    return test


def show(rows: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * 100)
    if rows.empty:
        print("  (no bouts matched)")
        return
    print(
        f"  {'date':<11}{'model pick':<24}{'conf':>6}{'market':>8}"
        f"  {'result':<8}{'actual winner':<24}"
    )
    print("-" * 100)
    for row in rows.sort_values("date").itertuples():
        market = (
            f"{row.market_prob_pick:.0%}" if pd.notna(row.market_prob_pick) else "  --"
        )
        mark = "HIT " if row.model_right else "miss"
        print(
            f"  {str(row.date.date()):<11}{row.pick[:23]:<24}{row.pick_prob:>5.0%}"
            f"{market:>8}  {mark:<8}{row.winner[:23]:<24}"
        )
    hits = rows.model_right.sum()
    print(f"\n  model: {hits}/{len(rows)} correct ({hits / len(rows):.0%})")
    scored = rows.dropna(subset=["market_right"])
    if len(scored):
        print(
            f"  market on the same bouts: "
            f"{int(scored.market_right.sum())}/{len(scored)} "
            f"({scored.market_right.mean():.0%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fighter", help="show every held-out bout for one fighter")
    parser.add_argument("--event", help="filter by event name, e.g. 'UFC 300'")
    parser.add_argument("--upsets", action="store_true",
                        help="underdogs the model correctly picked against the market")
    parser.add_argument("--disagreements", action="store_true",
                        help="where the model and the market differed most")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    test = build_table()
    print(
        f"Held-out test split: {len(test)} bouts, "
        f"{test.date.min().date()} to {test.date.max().date()} "
        "(none used in training)"
    )

    if args.fighter:
        key = normalize_name(args.fighter)
        rows = test[(test.a_key.str.contains(key, na=False)) | (test.b_key.str.contains(key, na=False))]
        show(rows, f"Held-out bouts involving {args.fighter}")
    elif args.event:
        rows = test[test.event_name.str.contains(args.event, case=False, na=False)]
        show(rows, f"Bouts matching event {args.event!r}")
    elif args.upsets:
        rows = test[
            test.model_right
            & (test.market_prob_pick < 0.45)
            & test.market_prob_pick.notna()
        ].nsmallest(args.n, "market_prob_pick")
        show(rows, "Underdogs the model called correctly (market gave them <45%)")
        print(
            "\n  NOTE: this view selects bouts *because* the model got them right, so the\n"
            "  hit rate here is 100% by construction and is not evidence of skill. It is\n"
            "  for seeing what kind of underdog the model likes. For an unbiased read,\n"
            "  use --disagreements or scripts/../mma_predictor/evaluate.py."
        )
    elif args.disagreements:
        rows = test[test.market_prob_pick.notna()].nlargest(args.n, "disagreement")
        show(rows, f"Largest disagreements with the closing line")
    else:
        rows = test.nlargest(args.n, "date")
        show(rows, f"Most recent {args.n} held-out bouts")


if __name__ == "__main__":
    main()
