"""Evaluate the model against every baseline that matters, including the market.

Accuracy alone is close to useless for a fight model: a probability of 0.51 and
a probability of 0.95 score identically if the favourite wins. What matters is
whether the numbers are *calibrated* -- when the model says 65%, does that
fighter win 65% of the time? -- and whether they carry information the closing
betting line does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .config import ARTIFACT_DIR, FEATURES_CSV, SplitConfig
from .dataset import prepare
from .features import load_features
from .parsing import american_to_decimal
from .train import ensemble_probabilities, load_checkpoint, to_tensors


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual win rate, bucketed. The heart of an honest report."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "actual": float(y[mask].mean()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def betting_simulation(
    frame: pd.DataFrame, p: np.ndarray, edge_threshold: float = 0.05, stake: float = 100.0
) -> dict[str, float]:
    """Flat-stake bets wherever the model disagrees enough with the closing line.

    This is the only test that asks the question a bettor cares about: does the
    model know something the market does not? It is deliberately unforgiving --
    bets are priced at the real closing line, vig included.
    """
    have = frame["market_prob_a"].notna().to_numpy()
    if not have.any():
        return {"n_bets": 0}
    y = frame["label"].to_numpy(float)[have]
    market = frame["market_prob_a"].to_numpy(float)[have]
    model = p[have]
    decimal_a = frame["a_odds"].map(american_to_decimal).to_numpy(float)[have]
    decimal_b = frame["b_odds"].map(american_to_decimal).to_numpy(float)[have]

    # Bet side A when the model likes A more than the market does, and vice versa.
    edge_a = model - market
    bet_a = edge_a > edge_threshold
    bet_b = -edge_a > edge_threshold
    placed = bet_a | bet_b
    if not placed.any():
        return {"n_bets": 0}

    won = np.where(bet_a, y > 0.5, y < 0.5)
    price = np.where(bet_a, decimal_a, decimal_b)
    profit = np.where(won, (price - 1.0) * stake, -stake)
    profit = profit[placed & np.isfinite(price)]
    staked = stake * len(profit)
    return {
        "edge_threshold": edge_threshold,
        "n_bets": int(len(profit)),
        "win_rate": float(won[placed & np.isfinite(price)].mean()),
        "total_profit": float(profit.sum()),
        "roi": float(profit.sum() / staked) if staked else float("nan"),
    }


def betting_controls(frame: pd.DataFrame, stake: float = 100.0) -> list[tuple[str, dict]]:
    """Reference strategies that make the model's betting result readable.

    Without these, a negative ROI is impossible to interpret. Backing every
    favourite should land near break-even (MMA markets carry a well-known
    favourite-longshot bias), backing every underdog should lose heavily, and
    betting at random should lose roughly the vig.
    """
    have = frame["market_prob_a"].notna().to_numpy()
    y = frame["label"].to_numpy(float)[have]
    market = frame["market_prob_a"].to_numpy(float)[have]
    decimal_a = frame["a_odds"].map(american_to_decimal).to_numpy(float)[have]
    decimal_b = frame["b_odds"].map(american_to_decimal).to_numpy(float)[have]

    def run(bet_a: np.ndarray) -> dict:
        price = np.where(bet_a, decimal_a, decimal_b)
        ok = np.isfinite(price)
        won = np.where(bet_a, y > 0.5, y < 0.5)[ok]
        profit = np.where(won, (price[ok] - 1.0) * stake, -stake)
        return {
            "n_bets": int(ok.sum()),
            "win_rate": float(won.mean()),
            "roi": float(profit.sum() / (stake * ok.sum())),
        }

    rng = np.random.default_rng(0)
    return [
        ("always back the favourite", run(market > 0.5)),
        ("always back the underdog", run(market < 0.5)),
        ("bet a random side", run(rng.random(len(y)) < 0.5)),
    ]


def permutation_importance(
    models, batch: dict[str, torch.Tensor], feature_names: list[str], repeats: int = 3
) -> pd.DataFrame:
    """How much does log-loss worsen when one feature is shuffled?

    The same column is shuffled in both slots with the same permutation, so the
    fighters stay comparable and only the feature's information is destroyed.
    """
    y = batch["y"].numpy()
    base = log_loss(y, ensemble_probabilities(models, batch), labels=[0, 1])
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(feature_names):
        deltas = []
        for _ in range(repeats):
            shuffled = {k: v.clone() for k, v in batch.items()}
            order = rng.permutation(len(y))
            shuffled["xa"][:, i] = batch["xa"][order, i]
            shuffled["xb"][:, i] = batch["xb"][order, i]
            deltas.append(
                log_loss(y, ensemble_probabilities(models, shuffled), labels=[0, 1]) - base
            )
        rows.append({"feature": name, "log_loss_increase": float(np.mean(deltas))})
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False)


def report(
    checkpoint: Path = ARTIFACT_DIR / "model.pt",
    features_path: Path = FEATURES_CSV,
    importance: bool = True,
) -> dict:
    models, transform, payload = load_checkpoint(checkpoint)
    splits = SplitConfig(**payload["config"]["splits"])
    frame = load_features(features_path)
    train_arrays, _, test_arrays, _, folds = prepare(
        frame, splits, use_market_odds=transform.use_market_odds
    )
    test_batch = to_tensors(test_arrays)
    test_frame = folds["test"]
    y = test_batch["y"].numpy()
    model_p = ensemble_probabilities(models, test_batch)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 78)
    emit(f"TEST SET: {len(y)} bouts from {test_frame.date.min().date()} "
         f"to {test_frame.date.max().date()}")
    emit("=" * 78)

    rows = [("Coin flip", metrics(y, np.full(len(y), 0.5)))]
    elo_p = 1.0 / (
        1.0 + 10.0 ** ((test_frame["b_elo"].to_numpy() - test_frame["a_elo"].to_numpy()) / 400.0)
    )
    rows.append(("Elo formula only", metrics(y, elo_p)))

    linear = LogisticRegression(C=0.01, max_iter=4000, fit_intercept=False).fit(
        train_arrays["xa"] - train_arrays["xb"], train_arrays["y"]
    )
    linear_p = linear.predict_proba(test_arrays["xa"] - test_arrays["xb"])[:, 1]
    rows.append(("Logistic regression", metrics(y, linear_p)))
    rows.append(("Neural network (ours)", metrics(y, model_p)))

    emit("")
    emit(f"{'Model':<26}{'n':>6}{'acc':>9}{'log loss':>11}{'Brier':>9}{'AUC':>8}")
    emit("-" * 78)
    for name, m in rows:
        emit(
            f"{name:<26}{m['n']:>6}{m['accuracy']:>9.4f}"
            f"{m['log_loss']:>11.4f}{m['brier']:>9.4f}{m['auc']:>8.4f}"
        )

    have_market = test_frame["market_prob_a"].notna().to_numpy()
    market_block: dict = {}
    if have_market.any():
        ym, pm = y[have_market], test_frame["market_prob_a"].to_numpy(float)[have_market]
        market_metrics = metrics(ym, pm)
        ours_same_rows = metrics(ym, model_p[have_market])
        emit("-" * 78)
        emit(f"{'Betting market (close)':<26}{market_metrics['n']:>6}"
             f"{market_metrics['accuracy']:>9.4f}{market_metrics['log_loss']:>11.4f}"
             f"{market_metrics['brier']:>9.4f}{market_metrics['auc']:>8.4f}")
        emit(f"{'  ...ours, same bouts':<26}{ours_same_rows['n']:>6}"
             f"{ours_same_rows['accuracy']:>9.4f}{ours_same_rows['log_loss']:>11.4f}"
             f"{ours_same_rows['brier']:>9.4f}{ours_same_rows['auc']:>8.4f}")
        market_block = {"market": market_metrics, "ours_on_market_rows": ours_same_rows}

    emit("")
    emit("CALIBRATION (does 65% mean 65%?)")
    table = calibration_table(y, model_p)
    emit(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ece = expected_calibration_error(y, model_p)
    emit(f"Expected calibration error: {ece:.4f}")

    emit("")
    emit("BETTING TEST vs the closing line (flat $100 stakes, real prices)")
    controls = []
    if have_market.any():
        controls = betting_controls(test_frame)
        for name, result in controls:
            emit(
                f"  [control] {name:<28} {result['n_bets']:4d} bets  "
                f"win rate {result['win_rate']:.3f}  ROI {result['roi']:+.2%}"
            )
        emit("")
    bets = {}
    for threshold in (0.03, 0.05, 0.08, 0.12):
        result = betting_simulation(test_frame, model_p, edge_threshold=threshold)
        bets[threshold] = result
        if result.get("n_bets"):
            emit(
                f"  edge > {threshold:.0%}: {result['n_bets']:4d} bets  "
                f"win rate {result['win_rate']:.3f}  "
                f"ROI {result['roi']:+.2%}  (${result['total_profit']:+,.0f})"
            )
        else:
            emit(f"  edge > {threshold:.0%}: no qualifying bets")

    if have_market.any():
        market_p = test_frame["market_prob_a"].to_numpy(float)[have_market]
        edge = model_p[have_market] - market_p
        qualifying = np.abs(edge) > 0.05
        if qualifying.any():
            on_underdog = ((edge > 0) == (market_p < 0.5))[qualifying].mean()
            emit(
                f"  -> {on_underdog:.0%} of our edge>5% bets are on the market's underdog."
            )

    importance_table = pd.DataFrame()
    if importance:
        emit("")
        emit("TOP FEATURES (permutation importance, test set)")
        importance_table = permutation_importance(models, test_batch, transform.feature_names)
        for row in importance_table.head(12).itertuples(index=False):
            emit(f"  {row.feature:<26} +{row.log_loss_increase:.5f} log loss when shuffled")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "evaluation_report.txt").write_text("\n".join(lines) + "\n")
    results = {
        "baselines": {name: m for name, m in rows},
        **market_block,
        "calibration": table.to_dict("records"),
        "ece": ece,
        "betting": {str(k): v for k, v in bets.items()},
        "betting_controls": {name: result for name, result in controls},
        "importance": importance_table.to_dict("records"),
    }
    (ARTIFACT_DIR / "evaluation.json").write_text(json.dumps(results, indent=1, default=float))
    emit("")
    emit(f"Wrote {ARTIFACT_DIR / 'evaluation_report.txt'} and evaluation.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_DIR / "model.pt")
    parser.add_argument("--no-importance", action="store_true")
    args = parser.parse_args()
    report(checkpoint=args.checkpoint, importance=not args.no_importance)


if __name__ == "__main__":
    main()
