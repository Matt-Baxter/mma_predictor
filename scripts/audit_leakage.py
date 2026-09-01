#!/usr/bin/env python
"""Reproduce the leakage audit that shaped this project's feature set.

Run it to see, from the data itself, why the pre-aggregated "average significant
strikes landed" columns are not model inputs here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from mma_predictor.features import ROLLING_FEATURES, load_features


def main() -> None:
    frame = load_features()
    both_debut = frame[(frame.a_is_ufc_debut == 1) & (frame.b_is_ufc_debut == 1)]
    print(__doc__)
    print(f"Bouts where BOTH fighters are making their UFC debut: {len(both_debut)}")
    print(
        "For these bouts neither fighter has a previous UFC fight, so every\n"
        '"average over previous fights" below must be empty. Any predictive\n'
        "power in them is information from inside the fight. 0.50 means the\n"
        "column tells you nothing about who won, which is what we should see.\n"
    )
    print(f"{'column':<26}{'populated':>11}{'predicts winner':>17}")
    print("-" * 54)
    for name in ROLLING_FEATURES:
        diff = both_debut[f"audit_a_{name}"].fillna(0) - both_debut[f"audit_b_{name}"].fillna(0)
        populated = int((diff != 0).sum())
        auc = roc_auc_score(both_debut.label, diff) if diff.std() > 0 else float("nan")
        print(f"{name:<26}{populated:>11}{auc:>17.3f}")

    print("\nFor comparison, the career features this project computes itself:")
    print(f"{'column':<26}{'populated':>11}{'predicts winner':>17}")
    print("-" * 54)
    for name in ("elo", "ufc_fights", "win_rate", "form_last5"):
        diff = both_debut[f"a_{name}"].fillna(0) - both_debut[f"b_{name}"].fillna(0)
        populated = int((diff != 0).sum())
        auc = roc_auc_score(both_debut.label, diff) if diff.std() > 0 else float("nan")
        print(f"{name:<26}{populated:>11}{'n/a (all zero)' if np.isnan(auc) else f'{auc:.3f}':>17}")
    print("\nAll zero, as they must be: a debutant has no UFC history to summarise.")


if __name__ == "__main__":
    main()
