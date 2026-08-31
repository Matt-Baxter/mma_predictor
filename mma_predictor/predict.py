"""Predict a hypothetical bout from two fighter names.

    python -m mma_predictor.predict "Israel Adesanya" "Sean Strickland"

Career state is replayed from the snapshot written by the feature build, so a
prediction uses exactly the same feature code as training did.
"""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ARTIFACT_DIR, FIGHTER_STATE_JSON, RAW_DIR
from .features import (
    CONTEXT_NUMERIC,
    FIGHTER_FEATURES,
    FighterState,
    RankingLookup,
    _static_features,
)
from .parsing import normalize_name, probability_to_american
from .train import ensemble_probabilities, load_checkpoint

DEFAULT_CHECKPOINT = ARTIFACT_DIR / "model.pt"


class FighterNotFound(LookupError):
    def __init__(self, name: str, suggestions: list[str]):
        self.suggestions = suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"No UFC fighter matching {name!r} in the dataset.{hint}")


@dataclass
class Matchup:
    a_name: str
    b_name: str
    probability_a: float
    weight_class: str
    scheduled_rounds: int
    title: bool
    as_of: pd.Timestamp

    @property
    def probability_b(self) -> float:
        return 1.0 - self.probability_a

    def render(self) -> str:
        pa, pb = self.probability_a, self.probability_b
        favourite = self.a_name if pa >= pb else self.b_name
        width = max(len(self.a_name), len(self.b_name))
        bar_a = "#" * int(round(pa * 40))
        bar_b = "#" * int(round(pb * 40))
        lines = [
            "",
            f"  {self.a_name} vs {self.b_name}",
            f"  {self.weight_class}, {self.scheduled_rounds} rounds"
            f"{', title bout' if self.title else ''} | as of {self.as_of.date()}",
            "",
            f"  {self.a_name:<{width}}  {pa:6.1%}  {probability_to_american(pa):+7.0f}  {bar_a}",
            f"  {self.b_name:<{width}}  {pb:6.1%}  {probability_to_american(pb):+7.0f}  {bar_b}",
            "",
            f"  Model favourite: {favourite} ({max(pa, pb):.1%})",
            "  Odds shown are fair (no vig) and are the model's own estimate,",
            "  produced without any bookmaker line as an input.",
            "",
        ]
        return "\n".join(lines)


class Predictor:
    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        snapshot_path: Path = FIGHTER_STATE_JSON,
        raw_dir: Path = RAW_DIR,
    ):
        self.models, self.transform, self.payload = load_checkpoint(checkpoint)
        if self.transform.use_market_odds:
            raise ValueError(
                "This checkpoint expects a bookmaker line as an input and cannot "
                "score a hypothetical bout. Use artifacts/model.pt instead."
            )
        self.snapshot = json.loads(Path(snapshot_path).read_text())
        self._display = {key: value["display_name"] for key, value in self.snapshot.items()}

        rankings = pd.read_csv(raw_dir / "ufc_rankings_dataset.csv")
        rankings = rankings.assign(
            date=pd.to_datetime(rankings["date"], errors="coerce"),
            name_key=rankings["fighter"].map(normalize_name),
        ).dropna(subset=["date", "rank"])
        self.rankings = RankingLookup(rankings)

    # ------------------------------------------------------------- lookup
    def resolve(self, name: str) -> str:
        key = normalize_name(name)
        if key in self.snapshot:
            return key
        matches = difflib.get_close_matches(key, list(self.snapshot), n=5, cutoff=0.75)
        if len(matches) == 1:
            return matches[0]
        if matches:
            # A unique prefix match beats a fuzzy tie ("jones" -> "jon jones").
            starts = [m for m in self.snapshot if m.startswith(key)]
            if len(starts) == 1:
                return starts[0]
        starts = [m for m in self.snapshot if key and key in m]
        if len(starts) == 1:
            return starts[0]
        raise FighterNotFound(name, [self._display[m] for m in (matches or starts)[:5]])

    def search(self, term: str, limit: int = 20) -> list[str]:
        key = normalize_name(term)
        hits = [self._display[k] for k in sorted(self.snapshot) if key in k]
        return hits[:limit]

    # ------------------------------------------------------------ scoring
    def _side(self, key: str, as_of: pd.Timestamp, weight_class: str) -> dict[str, float]:
        entry = self.snapshot[key]
        state = FighterState.from_json(entry["state"])
        features = state.career_features(as_of, weight_class)
        static = dict(entry["static"])
        static["dob"] = pd.to_datetime(static.get("dob")) if static.get("dob") else pd.NaT
        for numeric in ("height_in", "reach_in", "listed_weight"):
            value = static.get(numeric)
            static[numeric] = float(value) if value not in (None, "") else np.nan
        features.update(_static_features(static, as_of))
        features.update(self.rankings.features(key, weight_class, as_of))
        return features

    def predict(
        self,
        name_a: str,
        name_b: str,
        weight_class: str | None = None,
        scheduled_rounds: int | None = None,
        title: bool = False,
        as_of: str | pd.Timestamp | None = None,
    ) -> Matchup:
        key_a, key_b = self.resolve(name_a), self.resolve(name_b)
        if key_a == key_b:
            raise ValueError("A fighter cannot fight themselves.")
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
        weight_class = weight_class or self.snapshot[key_a]["last_weight_class"]
        scheduled_rounds = scheduled_rounds or (5 if title else 3)

        side_a = self._side(key_a, as_of_ts, weight_class)
        side_b = self._side(key_b, as_of_ts, weight_class)
        row: dict = {
            "weight_class": weight_class,
            "scheduled_rounds": float(scheduled_rounds),
            "is_title": 1.0 if title else 0.0,
            "is_womens": 1.0 if "women" in weight_class.lower() else 0.0,
            "label": 0.0,
            "a_stance": self.snapshot[key_a]["static"].get("stance", "Unknown"),
            "b_stance": self.snapshot[key_b]["static"].get("stance", "Unknown"),
        }
        for name in FIGHTER_FEATURES:
            row[f"a_{name}"] = side_a.get(name, np.nan)
            row[f"b_{name}"] = side_b.get(name, np.nan)
        for name in CONTEXT_NUMERIC:
            row.setdefault(name, np.nan)

        frame = pd.DataFrame([row])
        arrays = self.transform.transform(frame)
        batch = {k: torch.as_tensor(np.array(v)) for k, v in arrays.items()}
        probability = float(ensemble_probabilities(self.models, batch)[0])
        return Matchup(
            a_name=self._display[key_a],
            b_name=self._display[key_b],
            probability_a=probability,
            weight_class=weight_class,
            scheduled_rounds=int(scheduled_rounds),
            title=title,
            as_of=as_of_ts,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fighter_a", nargs="?", help="first fighter's name")
    parser.add_argument("fighter_b", nargs="?", help="second fighter's name")
    parser.add_argument("--weight-class", default=None)
    parser.add_argument("--rounds", type=int, default=None, help="3 or 5")
    parser.add_argument("--title", action="store_true", help="championship bout")
    parser.add_argument("--as-of", default=None, help="score the bout as of this date")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--search", default=None, help="list fighters matching a string")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    predictor = Predictor(checkpoint=args.checkpoint)
    if args.search:
        for name in predictor.search(args.search):
            print(name)
        return
    if not args.fighter_a or not args.fighter_b:
        parser.error("give two fighter names, or use --search")

    try:
        matchup = predictor.predict(
            args.fighter_a,
            args.fighter_b,
            weight_class=args.weight_class,
            scheduled_rounds=args.rounds,
            title=args.title,
            as_of=args.as_of,
        )
    except (FighterNotFound, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.json:
        print(
            json.dumps(
                {
                    "fighter_a": matchup.a_name,
                    "fighter_b": matchup.b_name,
                    "prob_a": round(matchup.probability_a, 4),
                    "prob_b": round(matchup.probability_b, 4),
                    "fair_odds_a": round(probability_to_american(matchup.probability_a)),
                    "fair_odds_b": round(probability_to_american(matchup.probability_b)),
                    "weight_class": matchup.weight_class,
                    "as_of": str(matchup.as_of.date()),
                },
                indent=1,
            )
        )
    else:
        print(matchup.render())


if __name__ == "__main__":
    main()
