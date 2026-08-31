"""Train the symmetric fight network and save a checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from .config import ARTIFACT_DIR, FEATURES_CSV, SplitConfig, TrainConfig
from .dataset import FeatureTransform, prepare
from .features import load_features
from .model import SymmetricFightNet

BATCH_KEYS = ("xa", "xb", "sa", "sb", "xc", "wc")


def to_tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    # np.array() forces a writable copy; pandas-derived arrays can be read-only.
    return {key: torch.as_tensor(np.array(value)) for key, value in arrays.items()}


def _forward(model: SymmetricFightNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(*(batch[key] for key in BATCH_KEYS))


@torch.no_grad()
def evaluate_split(model: SymmetricFightNet, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    probabilities = torch.sigmoid(_forward(model, batch)).numpy()
    y = batch["y"].numpy()
    return {
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "accuracy": float(((probabilities > 0.5) == (y > 0.5)).mean()),
        "auc": float(roc_auc_score(y, probabilities)) if len(set(y.tolist())) > 1 else float("nan"),
        "brier": float(np.mean((probabilities - y) ** 2)),
    }


def train_one(
    train_batch: dict[str, torch.Tensor],
    val_batch: dict[str, torch.Tensor],
    transform: FeatureTransform,
    config: TrainConfig,
    seed: int,
    verbose: bool = True,
) -> tuple[SymmetricFightNet, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SymmetricFightNet(
        n_fighter_features=transform.n_fighter_features,
        n_context_features=transform.n_context_features,
        n_weight_classes=len(transform.weight_class_vocab),
        n_stances=len(transform.stance_vocab),
        hidden_fighter=tuple(config.hidden_fighter),
        hidden_pair=tuple(config.hidden_pair),
        embed_dim=config.embed_dim,
        dropout=config.dropout,
        linear_skip=config.linear_skip,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    criterion = nn.BCEWithLogitsLoss()

    n = train_batch["y"].shape[0]
    eps = config.label_smoothing
    best_loss, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            index = order[start : start + config.batch_size]
            batch = {key: train_batch[key][index] for key in BATCH_KEYS}
            targets = train_batch["y"][index]
            # Label smoothing: no single fight deserves a 0/1 target when the
            # best possible model tops out well short of certainty.
            smoothed = targets * (1 - 2 * eps) + eps
            loss = criterion(_forward(model, batch), smoothed)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        scheduler.step()

        metrics = evaluate_split(model, val_batch)
        if metrics["log_loss"] < best_loss - 1e-5:
            best_loss = metrics["log_loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if verbose and (epoch % 20 == 0 or epoch == config.epochs - 1):
            print(
                f"    epoch {epoch:3d}  val_logloss={metrics['log_loss']:.4f}"
                f"  val_acc={metrics['accuracy']:.4f}  best={best_loss:.4f}@{best_epoch}"
            )
        if epoch - best_epoch >= config.patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (best {best_loss:.4f} @ {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_log_loss": best_loss, "best_epoch": best_epoch}


def train(
    features_path: Path = FEATURES_CSV,
    config: TrainConfig | None = None,
    n_models: int = 5,
    output: Path | None = None,
) -> Path:
    config = config or TrainConfig()
    frame = load_features(features_path)
    train_arrays, val_arrays, test_arrays, transform, folds = prepare(
        frame, config.splits, use_market_odds=config.use_market_odds
    )
    train_batch = to_tensors(train_arrays)
    val_batch = to_tensors(val_arrays)
    test_batch = to_tensors(test_arrays)

    print(
        f"Splits  train={len(folds['train'])}  val={len(folds['val'])}  test={len(folds['test'])}"
        f"  |  {transform.n_fighter_features} features per fighter"
    )
    print(f"Market odds used as an input: {config.use_market_odds}")

    models, histories = [], []
    for i in range(n_models):
        print(f"  model {i + 1}/{n_models} (seed {config.seed + i})")
        model, history = train_one(
            train_batch, val_batch, transform, config, seed=config.seed + i
        )
        models.append(model)
        histories.append(history)

    ensemble_val = ensemble_metrics(models, val_batch)
    ensemble_test = ensemble_metrics(models, test_batch)
    print(f"\nEnsemble  val: {_fmt(ensemble_val)}")
    print(f"Ensemble test: {_fmt(ensemble_test)}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = output or (
        ARTIFACT_DIR / ("model_with_odds.pt" if config.use_market_odds else "model.pt")
    )
    torch.save(
        {
            "state_dicts": [m.state_dict() for m in models],
            "transform": transform.to_json(),
            "config": {
                **{k: v for k, v in asdict(config).items() if k != "splits"},
                "splits": asdict(config.splits),
            },
            "architecture": {
                "n_fighter_features": transform.n_fighter_features,
                "n_context_features": transform.n_context_features,
                "n_weight_classes": len(transform.weight_class_vocab),
                "n_stances": len(transform.stance_vocab),
                "hidden_fighter": list(config.hidden_fighter),
                "hidden_pair": list(config.hidden_pair),
                "embed_dim": config.embed_dim,
                "dropout": config.dropout,
                "linear_skip": config.linear_skip,
            },
            "histories": histories,
            "metrics": {"val": ensemble_val, "test": ensemble_test},
        },
        output,
    )
    print(f"\nSaved checkpoint -> {output}")
    return output


@torch.no_grad()
def ensemble_probabilities(
    models: list[SymmetricFightNet], batch: dict[str, torch.Tensor]
) -> np.ndarray:
    """Average the member probabilities. Averaging preserves antisymmetry."""
    stacked = []
    for model in models:
        model.eval()
        stacked.append(torch.sigmoid(_forward(model, batch)).numpy())
    return np.mean(stacked, axis=0)


def ensemble_metrics(
    models: list[SymmetricFightNet], batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    probabilities = ensemble_probabilities(models, batch)
    y = batch["y"].numpy()
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "accuracy": float(((probabilities > 0.5) == (y > 0.5)).mean()),
        "auc": float(roc_auc_score(y, probabilities)),
        "brier": float(np.mean((probabilities - y) ** 2)),
    }


def load_checkpoint(path: Path) -> tuple[list[SymmetricFightNet], FeatureTransform, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    transform = FeatureTransform.from_json(payload["transform"])
    models = []
    for state in payload["state_dicts"]:
        model = SymmetricFightNet(**payload["architecture"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, transform, payload


def _fmt(metrics: dict[str, float]) -> str:
    return "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-models", type=int, default=5, help="ensemble size")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--use-market-odds",
        action="store_true",
        help="train the market-aware variant (for comparison only - see README)",
    )
    parser.add_argument(
        "--no-linear-skip",
        action="store_true",
        help="ablate the linear skip connection (see README)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        dropout=args.dropout,
        lr=args.lr,
        seed=args.seed,
        use_market_odds=args.use_market_odds,
        linear_skip=not args.no_linear_skip,
        splits=SplitConfig(),
    )
    train(config=config, n_models=args.n_models, output=args.output)


if __name__ == "__main__":
    main()
