"""Turn the feature table into model-ready tensors, with honest preprocessing.

Two rules are enforced here:

1. **Chronological splits.** Train on the past, validate on the near past, test
   on the most recent fights. A random split would put a fighter's May bout in
   train and their November bout in test, and the two rows share most of their
   career history.
2. **Statistics fitted on train only**, and fitted on slots A and B *pooled*.
   The two slots hold the same quantities, so they must share one mean, one
   standard deviation and one median -- otherwise the preprocessing itself
   would break the symmetry the model is built around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SplitConfig
from .features import CONTEXT_NUMERIC, FIGHTER_FEATURES

UNKNOWN = "<unk>"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class FeatureTransform:
    """Imputation + standardisation + categorical vocabularies, fitted on train."""

    feature_names: list[str]
    context_names: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    context_means: dict[str, float]
    context_stds: dict[str, float]
    stance_vocab: list[str]
    weight_class_vocab: list[str]
    use_market_odds: bool = False

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(
        cls, train: pd.DataFrame, use_market_odds: bool = False
    ) -> "FeatureTransform":
        names = list(FIGHTER_FEATURES)
        if use_market_odds:
            names.append("market_logit")
        medians, means, stds = {}, {}, {}
        for name in names:
            pooled = np.concatenate(
                [
                    _column(train, "a", name).to_numpy(dtype=float),
                    _column(train, "b", name).to_numpy(dtype=float),
                ]
            )
            median = float(np.nanmedian(pooled)) if np.isfinite(pooled).any() else 0.0
            medians[name] = median
            filled = np.where(np.isnan(pooled), median, pooled)
            means[name] = float(filled.mean())
            std = float(filled.std())
            stds[name] = std if std > 1e-8 else 1.0

        context_means, context_stds = {}, {}
        for name in CONTEXT_NUMERIC:
            values = train[name].to_numpy(dtype=float)
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            filled = np.where(np.isnan(values), median, values)
            medians[f"ctx::{name}"] = median
            context_means[name] = float(filled.mean())
            std = float(filled.std())
            context_stds[name] = std if std > 1e-8 else 1.0

        stances = sorted(
            set(train["a_stance"].astype(str)) | set(train["b_stance"].astype(str))
        )
        weight_classes = sorted(set(train["weight_class"].astype(str)))
        return cls(
            feature_names=names,
            context_names=list(CONTEXT_NUMERIC),
            medians=medians,
            means=means,
            stds=stds,
            context_means=context_means,
            context_stds=context_stds,
            stance_vocab=[UNKNOWN, *stances],
            weight_class_vocab=[UNKNOWN, *weight_classes],
            use_market_odds=use_market_odds,
        )

    # ------------------------------------------------------------- applying
    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        def side(prefix: str) -> np.ndarray:
            columns = []
            for name in self.feature_names:
                values = _column(frame, prefix, name).to_numpy(dtype=float)
                values = np.where(np.isnan(values), self.medians[name], values)
                columns.append((values - self.means[name]) / self.stds[name])
            return np.stack(columns, axis=1).astype(np.float32)

        context = []
        for name in self.context_names:
            values = frame[name].to_numpy(dtype=float)
            values = np.where(np.isnan(values), self.medians[f"ctx::{name}"], values)
            context.append((values - self.context_means[name]) / self.context_stds[name])
        context_array = np.stack(context, axis=1).astype(np.float32)

        stance_index = {name: i for i, name in enumerate(self.stance_vocab)}
        wc_index = {name: i for i, name in enumerate(self.weight_class_vocab)}
        return {
            "xa": side("a"),
            "xb": side("b"),
            "sa": frame["a_stance"].astype(str).map(lambda s: stance_index.get(s, 0)).to_numpy(np.int64),
            "sb": frame["b_stance"].astype(str).map(lambda s: stance_index.get(s, 0)).to_numpy(np.int64),
            "xc": context_array,
            "wc": frame["weight_class"].astype(str).map(lambda s: wc_index.get(s, 0)).to_numpy(np.int64),
            "y": frame["label"].to_numpy(np.float32),
        }

    @property
    def n_fighter_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_context_features(self) -> int:
        return len(self.context_names)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)

    @classmethod
    def from_json(cls, payload: str) -> "FeatureTransform":
        return cls(**json.loads(payload))


def _column(frame: pd.DataFrame, prefix: str, name: str) -> pd.Series:
    """Fetch ``a_<name>`` / ``b_<name>``, synthesising the market feature."""
    if name == "market_logit":
        market = frame["market_prob_a"].to_numpy(dtype=float)
        logits = _logit(market)
        logits = np.where(np.isnan(market), np.nan, logits)
        return pd.Series(logits if prefix == "a" else -logits, index=frame.index)
    return frame[f"{prefix}_{name}"]


def chronological_split(
    frame: pd.DataFrame, splits: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice the feature table into train / validation / test by fight date."""
    dates = pd.to_datetime(frame["date"])
    train = frame[
        (dates >= splits.train_start) & (dates < splits.val_start)
    ].reset_index(drop=True)
    val = frame[(dates >= splits.val_start) & (dates < splits.test_start)].reset_index(
        drop=True
    )
    test = frame[(dates >= splits.test_start) & (dates < splits.end)].reset_index(
        drop=True
    )
    return train, val, test


def prepare(
    frame: pd.DataFrame,
    splits: SplitConfig,
    use_market_odds: bool = False,
) -> tuple[dict, dict, dict, FeatureTransform, dict[str, pd.DataFrame]]:
    """Split, fit the transform on train alone, and encode all three folds."""
    frame = frame.dropna(subset=["label"]).copy()
    if use_market_odds:
        # A market-aware variant can only be trained where a closing line exists.
        frame = frame[frame["market_prob_a"].notna()].reset_index(drop=True)
    train, val, test = chronological_split(frame, splits)
    if len(train) == 0:
        raise ValueError("Empty training split - check SplitConfig dates.")
    transform = FeatureTransform.fit(train, use_market_odds=use_market_odds)
    return (
        transform.transform(train),
        transform.transform(val),
        transform.transform(test),
        transform,
        {"train": train, "val": val, "test": test},
    )
