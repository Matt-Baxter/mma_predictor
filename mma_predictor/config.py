"""Paths, dataset URLs and the knobs that control the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

# TidyTuesday 2026-07-07: "UFC Athletes and Fight Data", curated by Benjamin Smith
# from the {fightr} R package (UFCStats, UFC athlete profiles, Kaggle, Octagon API).
TIDYTUESDAY_BASE = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2026/2026-07-07"
)

DATASETS = (
    "ufc_athletes",
    "ufc_fights",
    "ufc_rankings_dataset",
    "ufcstats_data",
    "ultimate_ufc_dataset",
)

FEATURES_PARQUET = PROCESSED_DIR / "fight_features.parquet"
FEATURES_CSV = PROCESSED_DIR / "fight_features.csv"
FIGHTER_STATE_JSON = PROCESSED_DIR / "fighter_state.json"

# Elo settings. K is deliberately modest: UFC fighters average ~6 bouts each, so a
# large K makes ratings thrash on small samples.
ELO_START = 1500.0
ELO_K = 24.0
# Finishes move the rating more than decisions do; a split decision moves it less.
ELO_METHOD_MULTIPLIER = {
    "ko_tko": 1.30,
    "submission": 1.25,
    "decision_unanimous": 1.00,
    "decision_split": 0.80,
    "decision_majority": 0.90,
    "other": 1.00,
}


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/val/test boundaries.

    Fight prediction must never be split randomly: a fighter's features on
    2019-05-04 are built from the same career history as their features on
    2019-11-02, so a random split lets near-duplicate rows straddle the
    boundary and the test score becomes fiction.
    """

    train_start: str = "2001-01-01"
    val_start: str = "2021-01-01"
    test_start: str = "2023-06-01"
    end: str = "2100-01-01"


@dataclass
class TrainConfig:
    # Chosen by a validation sweep (scripts/sweep.py). Every configuration
    # tried landed within ~0.004 validation log-loss of every other, which is
    # inside the noise of a 1,205-fight validation set, so the smallest network
    # in that band wins on the one-standard-error rule: with ~5.5k training
    # bouts a wider net buys nothing but variance.
    hidden_fighter: tuple[int, ...] = (48,)
    hidden_pair: tuple[int, ...] = (48,)
    embed_dim: int = 8
    dropout: float = 0.35
    # Every learning rate from 2.5e-4 to 1.5e-3 landed within 0.002 validation
    # log-loss of the others. The slowest was kept: it converges over ~8 epochs
    # instead of peaking on the first one, which makes the curve worth reading.
    lr: float = 2.5e-4
    weight_decay: float = 1e-3
    batch_size: int = 256
    epochs: int = 250
    patience: int = 40
    label_smoothing: float = 0.03
    seed: int = 17
    use_market_odds: bool = False
    splits: SplitConfig = field(default_factory=SplitConfig)
