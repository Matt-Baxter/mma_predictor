#!/usr/bin/env python
"""Export the trained model, fighter states and market history as one JSON file.

The web page in `web/` runs the network in the browser, so everything it needs
travels in this file: the ensemble weights, the preprocessing statistics, each
fighter's pre-computed feature vector, and the closing lines for every bout the
two fighters have actually contested.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mma_predictor.config import ARTIFACT_DIR, FIGHTER_STATE_JSON, RAW_DIR, SplitConfig
from mma_predictor.dataset import prepare
from mma_predictor.features import (
    CONTEXT_NUMERIC,
    FIGHTER_FEATURES,
    FighterState,
    RankingLookup,
    _static_features,
    load_features,
)
from mma_predictor.parsing import DIVISION_WEIGHT, normalize_name
from mma_predictor.train import ensemble_probabilities, load_checkpoint, to_tensors

OUTPUT = Path(__file__).resolve().parent.parent / "web" / "model_data.json"


def round_list(values, digits=5):
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), digits)
            for v in values]


def export_weights(model) -> dict:
    """Pull the tensors out in the order the browser will replay them."""
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}

    def blocks(prefix: str) -> list[dict]:
        out, i = [], 0
        while f"{prefix}.{i}.weight" in state:
            out.append(
                {
                    "W": [round_list(row) for row in state[f"{prefix}.{i}.weight"]],
                    "b": round_list(state[f"{prefix}.{i}.bias"]),
                    "g": round_list(state[f"{prefix}.{i + 1}.weight"]),
                    "be": round_list(state[f"{prefix}.{i + 1}.bias"]),
                }
            )
            i += 4  # Linear, LayerNorm, SiLU, Dropout
        return out

    return {
        "wcEmb": [round_list(row) for row in state["weight_class_embedding.weight"]],
        "stEmb": [round_list(row) for row in state["stance_embedding.weight"]],
        "encoder": blocks("fighter_encoder"),
        "pair": blocks("pair_body"),
        "outW": round_list(state["pair_out.weight"][0]),
        "outB": float(state["pair_out.bias"][0]),
        # The linear skip branch, absent when the model was trained with
        # --no-linear-skip.
        "skipW": (
            round_list(state["linear_skip.weight"][0])
            if "linear_skip.weight" in state
            else None
        ),
    }


def main() -> None:
    models, transform, payload = load_checkpoint(ARTIFACT_DIR / "model.pt")
    frame = load_features()
    reference_date = pd.Timestamp(frame.date.max())
    print(f"Reference date (last event in the data): {reference_date.date()}")

    # ---- model probabilities for every historical bout, for the market panel
    splits = SplitConfig(**payload["config"]["splits"])
    arrays = transform.transform(frame)
    probabilities = ensemble_probabilities(models, to_tensors(arrays))
    frame = frame.assign(model_prob_a=probabilities)

    def split_of(date: pd.Timestamp) -> str:
        if date < pd.Timestamp(splits.val_start):
            return "train"
        if date < pd.Timestamp(splits.test_start):
            return "val"
        return "test"

    meetings: dict[str, list] = {}
    for row in frame.itertuples():
        key = "|".join(sorted((row.a_key, row.b_key)))
        meetings.setdefault(key, []).append(
            {
                "d": str(row.date.date()),
                "a": row.a_key,
                "ao": None if pd.isna(row.a_odds) else int(row.a_odds),
                "bo": None if pd.isna(row.b_odds) else int(row.b_odds),
                "mp": None if pd.isna(row.market_prob_a) else round(float(row.market_prob_a), 4),
                "w": row.a_key if row.label == 1 else row.b_key,
                "mm": round(float(row.model_prob_a), 4),
                "s": split_of(row.date),
                "m": row.method_class,
            }
        )

    # ---- per-fighter feature vectors as of the reference date
    snapshot = json.loads(FIGHTER_STATE_JSON.read_text())
    rankings = pd.read_csv(RAW_DIR / "ufc_rankings_dataset.csv")
    rankings = rankings.assign(
        date=pd.to_datetime(rankings["date"], errors="coerce"),
        name_key=rankings["fighter"].map(normalize_name),
    ).dropna(subset=["date", "rank"])
    lookup = RankingLookup(rankings)

    divisions = [d for d in transform.weight_class_vocab if d != "<unk>"]
    fighters = {}
    for key, entry in snapshot.items():
        state = FighterState.from_json(entry["state"])
        if state.fights == 0:
            continue  # never actually competed in the UFC
        last_division = entry["last_weight_class"]
        features = state.career_features(reference_date, last_division)
        static = dict(entry["static"])
        static["dob"] = pd.to_datetime(static.get("dob")) if static.get("dob") else pd.NaT
        for numeric in ("height_in", "reach_in", "listed_weight"):
            value = static.get(numeric)
            static[numeric] = float(value) if value not in (None, "") else np.nan
        features.update(_static_features(static, reference_date))
        features.update(lookup.features(key, last_division, reference_date))

        # Rank per division so the page can re-derive rank features for whatever
        # division the chosen matchup lands in.
        ranks = {}
        for division in divisions:
            rank = RankingLookup._as_of(lookup._div, (key, division), reference_date)
            if rank == rank:
                ranks[division] = int(rank)
        p4p = RankingLookup._as_of(lookup._p4p, key, reference_date)

        fighters[key] = {
            "n": entry["display_name"],
            "f": round_list([features.get(name, np.nan) for name in FIGHTER_FEATURES], 4),
            "dc": {k: int(v) for k, v in state.division_counts.items()},
            "ld": last_division,
            "rk": ranks,
            "p4p": None if p4p != p4p else int(p4p),
            "st": entry["static"].get("stance", "Unknown"),
            "rec": f"{state.wins}-{state.losses}" + (f"-{state.draws}" if state.draws else ""),
            "last": None if state.last_date is None else str(pd.Timestamp(state.last_date).date()),
        }

    bundle = {
        "meta": {
            "referenceDate": str(reference_date.date()),
            "testStart": splits.test_start,
            "valStart": splits.val_start,
            "nFighters": len(fighters),
            "nBouts": int(len(frame)),
        },
        "featureNames": list(FIGHTER_FEATURES),
        "means": round_list([transform.means[n] for n in transform.feature_names]),
        "stds": round_list([transform.stds[n] for n in transform.feature_names]),
        "medians": round_list([transform.medians[n] for n in transform.feature_names]),
        "contextNames": list(CONTEXT_NUMERIC),
        "ctxMeans": round_list([transform.context_means[n] for n in CONTEXT_NUMERIC]),
        "ctxStds": round_list([transform.context_stds[n] for n in CONTEXT_NUMERIC]),
        "ctxMedians": round_list([transform.medians[f"ctx::{n}"] for n in CONTEXT_NUMERIC]),
        "stanceVocab": transform.stance_vocab,
        "divisionVocab": transform.weight_class_vocab,
        "divisionWeight": DIVISION_WEIGHT,
        "models": [export_weights(m) for m in models],
        "fighters": fighters,
        "meetings": meetings,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(bundle, separators=(",", ":")))
    size = OUTPUT.stat().st_size
    print(f"Wrote {OUTPUT} ({size / 1e6:.2f} MB)")
    print(f"  {len(fighters)} fighters, {len(meetings)} matchups with history")


if __name__ == "__main__":
    main()
