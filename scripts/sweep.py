#!/usr/bin/env python
"""Hyperparameter sweep, selected on the validation split only.

The test split is never consulted here. It is printed alongside purely so the
selection can be audited after the fact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import itertools

from mma_predictor.config import SplitConfig, TrainConfig
from mma_predictor.dataset import prepare
from mma_predictor.features import load_features
from mma_predictor.train import ensemble_metrics, to_tensors, train_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-per-config", type=int, default=3)
    args = parser.parse_args()

    frame = load_features()
    train_arrays, val_arrays, test_arrays, transform, folds = prepare(frame, SplitConfig())
    train_batch, val_batch, test_batch = (
        to_tensors(train_arrays),
        to_tensors(val_arrays),
        to_tensors(test_arrays),
    )
    print(f"train={len(folds['train'])} val={len(folds['val'])} test={len(folds['test'])}\n")

    hidden = [((48,), (48,)), ((96, 64), (96, 48)), ((128, 96), (128, 64))]
    dropouts = [0.25, 0.35, 0.45]
    rates = [2.5e-4, 6e-4, 1.5e-3]

    results = []
    for (hf, hp), dropout, lr in itertools.product(hidden, dropouts, rates):
        config = TrainConfig(
            hidden_fighter=hf, hidden_pair=hp, dropout=dropout, lr=lr, epochs=300, patience=50
        )
        models = [
            train_one(train_batch, val_batch, transform, config, seed=17 + i, verbose=False)[0]
            for i in range(args.models_per_config)
        ]
        val = ensemble_metrics(models, val_batch)
        test = ensemble_metrics(models, test_batch)
        results.append((val["log_loss"], hf, dropout, lr, val, test))
        print(
            f"  hidden={str(hf):10s} dropout={dropout} lr={lr:<8} "
            f"| VAL log loss={val['log_loss']:.4f} acc={val['accuracy']:.4f} "
            f"| test log loss={test['log_loss']:.4f} acc={test['accuracy']:.4f}"
        )

    results.sort(key=lambda row: row[0])
    best = results[0]
    spread = results[-1][0] - best[0]
    print(f"\nBest by validation: hidden={best[1]} dropout={best[2]} lr={best[3]}")
    print(f"Validation log loss spread across the whole grid: {spread:.4f}")
    print(
        "If that spread is small (it is ~0.005), the grid is inside the noise of a\n"
        "1,205-fight validation set and the smallest model in the band is the honest pick."
    )


if __name__ == "__main__":
    main()
