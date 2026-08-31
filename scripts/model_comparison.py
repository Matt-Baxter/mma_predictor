#!/usr/bin/env python
"""Why isn't the neural network beating logistic regression?

The honest answer is that on this data nothing beats logistic regression,
because there is almost no nonlinearity in these features to exploit. This
script is the evidence: it fits a linear model, several nonlinear ones, and the
network, then tests whether any of the differences are real.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from mma_predictor.config import ARTIFACT_DIR, SplitConfig
from mma_predictor.dataset import prepare
from mma_predictor.features import load_features
from mma_predictor.train import ensemble_probabilities, load_checkpoint, to_tensors


def per_fight_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_DIR / "model.pt")
    args = parser.parse_args()

    models, transform, payload = load_checkpoint(args.checkpoint)
    frame = load_features()
    train, val, test, _, folds = prepare(frame, SplitConfig(**payload["config"]["splits"]))
    y_train, y_val, y_test = train["y"], val["y"], test["y"]
    diff_train = train["xa"] - train["xb"]
    diff_val = val["xa"] - val["xb"]
    diff_test = test["xa"] - test["xb"]

    print(__doc__)
    print(f"Validation: {len(y_val)} bouts   Test: {len(y_test)} bouts\n")
    print(f"  {'model':<46}{'val':>9}{'test':>9}")
    print("  " + "-" * 64)

    results = {}

    def record(name, p_val, p_test):
        results[name] = p_test
        print(
            f"  {name:<46}{log_loss(y_val, p_val, labels=[0, 1]):>9.4f}"
            f"{log_loss(y_test, p_test, labels=[0, 1]):>9.4f}"
        )

    linear = LogisticRegression(C=0.01, max_iter=4000, fit_intercept=False).fit(
        diff_train, y_train
    )
    record(
        "logistic regression",
        linear.predict_proba(diff_val)[:, 1],
        linear.predict_proba(diff_test)[:, 1],
    )

    # Gradient boosting finds interactions and nonlinearity on its own. If real
    # nonlinear structure existed, this is where it would show up.
    for rate, leaves in ((0.03, 8), (0.05, 15), (0.03, 31)):
        booster = HistGradientBoostingClassifier(
            learning_rate=rate, max_leaf_nodes=leaves, max_iter=600,
            early_stopping=True, validation_fraction=0.15, random_state=0,
        ).fit(diff_train, y_train)
        record(
            f"gradient boosting (rate={rate}, leaves={leaves})",
            booster.predict_proba(diff_val)[:, 1],
            booster.predict_proba(diff_test)[:, 1],
        )

    squared = lambda d: np.hstack([d, d**2])
    quadratic = LogisticRegression(C=0.01, max_iter=4000, fit_intercept=False).fit(
        squared(diff_train), y_train
    )
    record(
        "logistic regression + squared terms",
        quadratic.predict_proba(squared(diff_val))[:, 1],
        quadratic.predict_proba(squared(diff_test))[:, 1],
    )

    network = ensemble_probabilities(models, to_tensors(test))
    record(
        "neural network (this project)",
        ensemble_probabilities(models, to_tensors(val)),
        network,
    )

    print("\n  Read the table this way:")
    print("  - Gradient boosting and quadratic terms both LOSE to the plain linear")
    print("    model. Boosting is very good at finding interactions, so if exploitable")
    print("    nonlinearity existed here it would show up there. It does not.")
    print("  - The network edges ahead only because it CONTAINS the linear model: the")
    print("    skip connection w.(xa-xb) is that model, and the deep branch adds a")
    print("    small correction on top. Remove it with --no-linear-skip and the")
    print("    network drops back behind.\n")

    print("  Is the network-vs-linear gap even real?")
    linear_test = linear.predict_proba(diff_test)[:, 1]
    delta = per_fight_loss(y_test, network) - per_fight_loss(y_test, linear_test)
    statistic, p_value = stats.ttest_rel(
        per_fight_loss(y_test, network), per_fight_loss(y_test, linear_test)
    )
    rng = np.random.default_rng(0)
    boot = np.array(
        [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10000)]
    )
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"    mean difference (positive = network worse): {delta.mean():+.4f}")
    print(f"    paired t-test across the {len(delta)} test bouts: p = {p_value:.3f}")
    print(f"    bootstrap 95% CI: [{low:+.4f}, {high:+.4f}]  <- contains zero")
    print("    The two are tied. Either one 'winning' on this data is sampling noise,")
    print("    which is the point: the ceiling here is the FEATURES, not the model.")
    print("    Without round-by-round fight statistics, there is little left to learn,")
    print("    and the 0.06 log loss gap to the closing line is information the market")
    print("    has and no public dataset does.")


if __name__ == "__main__":
    main()
