#!/usr/bin/env python
"""Write PyTorch predictions for a sample of matchups, for the JS check to match."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mma_predictor.features import load_features
from mma_predictor.predict import Predictor

OUT = Path(__file__).resolve().parent.parent / "web" / "predictions.json"

PAIRS = [
    ("sean o malley", "umar nurmagomedov", False),
    ("jon jones", "stipe miocic", False),
    ("alex pereira", "tom aspinall", False),
    ("israel adesanya", "sean strickland", True),
    ("merab dvalishvili", "sean o malley", True),
    ("zhang weili", "rose namajunas", False),
    ("islam makhachev", "ilia topuria", True),
    ("jon jones", "alex pereira", False),
]


def main() -> None:
    predictor = Predictor()
    reference = str(pd.Timestamp(load_features().date.max()).date())
    keys = sorted(predictor.snapshot)
    random.Random(7).shuffle(keys)
    pairs = list(PAIRS)
    # Add random pairs so the check is not limited to hand-picked names.
    for i in range(0, 40, 2):
        pairs.append((keys[i], keys[i + 1], i % 4 == 0))

    rows = []
    for a, b, title in pairs:
        if a not in predictor.snapshot or b not in predictor.snapshot or a == b:
            continue
        matchup = predictor.predict(a, b, title=title, as_of=reference)
        rows.append({"a": a, "b": b, "title": title, "p": round(matchup.probability_a, 8)})
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {len(rows)} reference predictions to {OUT} (as of {reference})")


if __name__ == "__main__":
    main()
