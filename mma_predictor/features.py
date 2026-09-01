"""Build one leak-free, pre-fight feature row per UFC bout.

The whole file obeys a single rule: **a feature attached to a fight may only
use information that existed before that fight's opening horn.** Everything is
produced by one chronological pass over the bout history. Fights on the same
event are emitted before any of them updates the running state, so two fighters
on the same card can never see each other's result.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ELO_K,
    ELO_METHOD_MULTIPLIER,
    ELO_START,
    FEATURES_CSV,
    FIGHTER_STATE_JSON,
    PROCESSED_DIR,
    RAW_DIR,
)
from .parsing import (
    american_to_implied,
    parse_of_pair,
    classify_method,
    clean_weight_class,
    devig_two_way,
    is_title_bout,
    normalize_name,
    parse_clock_seconds,
    parse_height_inches,
    parse_inches,
    parse_percent,
    parse_scheduled_rounds,
    parse_weight_lbs,
)

# Fights before this date only warm up Elo and career counters; they are not
# modelled. 2001 is where the modern sport starts: the Unified Rules were
# adopted in Nov 2000, and the Zuffa era began in Jan 2001. Earlier bouts
# (open-weight, one-night tournaments, no judges) still feed the running
# career state, they just never become training rows.
MODEL_ERA_START = "2001-01-01"

# Career-history features emitted for each fighter. Kept in one place so the
# pair-wise `f1_*` / `f2_*` columns and the difference columns stay in sync.
CAREER_FEATURES = (
    "ufc_fights",
    "ufc_wins",
    "ufc_losses",
    "win_rate",
    "win_streak",
    "lose_streak",
    "best_win_streak",
    "form_last3",
    "form_last5",
    "ko_win_rate",
    "sub_win_rate",
    "dec_win_rate",
    "finish_rate",
    "ko_loss_rate",
    "sub_loss_rate",
    "finished_rate",
    "avg_fight_minutes",
    "total_octagon_minutes",
    "title_bouts",
    "elo",
    "elo_peak",
    "elo_off_peak",
    "opponent_elo_avg",
    "opponent_elo_last3",
    "elo_momentum",
    "quality_wins",
    "days_since_last_fight",
    "days_since_debut",
    "is_ufc_debut",
    "division_fights",
    "changed_division",
)

# In-fight performance, accumulated from round-by-round UFCStats data over the
# fighter's PREVIOUS bouts only. These are the honest versions of the columns
# rejected in the leakage audit: same quantities, computed here so the cutoff is
# ours to enforce.
IN_FIGHT_FEATURES = (
    "sig_str_landed_pm",
    "sig_str_absorbed_pm",
    "sig_str_accuracy",
    "sig_str_defense",
    "strike_differential_pm",
    "takedowns_per15",
    "takedown_accuracy",
    "takedown_defense",
    "sub_attempts_per15",
    "knockdowns_per15",
    "knockdowns_absorbed_per15",
    "control_share",
    "control_differential",
    "head_strike_share",
    "leg_strike_share",
    "ground_strike_share",
    "distance_strike_share",
)

STATIC_FEATURES = (
    "age",
    "age_from_prime",
    "height_in",
    "reach_in",
    "ape_index",
    "listed_weight",
)

RANK_FEATURES = (
    "div_rank_score",
    "is_champion",
    "is_ranked",
    "p4p_rank_score",
)

# Rolling in-fight statistics carried over from the Ultimate UFC dataset.
#
# These are ADVERTISED as averages over a fighter's previous UFC bouts, and it
# is tempting to use them: strikes landed and takedown accuracy are exactly the
# kind of thing a fight model wants. They are NOT model inputs here, because
# they fail an audit (see scripts/audit_leakage.py): in bouts where *both*
# fighters are making their UFC debut -- where every one of these numbers must
# be empty, since neither man has a previous UFC bout to average -- they are
# populated for a chunk of rows and predict the winner at AUC 0.56. That is
# information from inside the fight leaking backwards.
#
# They are still written to the processed CSV under an `audit_` prefix so the
# audit is reproducible, and they are excluded from FIGHTER_FEATURES below.
ROLLING_FEATURES = (
    "avg_sig_str_landed",
    "avg_sig_str_pct",
    "avg_sub_att",
    "avg_td_landed",
    "avg_td_pct",
)

# Model inputs. Every one of these is computed in this file from raw bout
# history, static fighter profiles, or as-of-date rankings -- nothing is taken
# from a pre-aggregated source we cannot verify.
FIGHTER_FEATURES = CAREER_FEATURES + IN_FIGHT_FEATURES + STATIC_FEATURES + RANK_FEATURES

# `era_year` is deliberately absent: it is stored for splitting and analysis but
# is not a model input. Every prediction we care about is for a *future* fight,
# so a raw calendar-time feature is always extrapolated beyond its training
# range -- the one input guaranteed to be out of distribution at inference.
CONTEXT_NUMERIC = ("scheduled_rounds", "is_title", "is_womens")
CONTEXT_CATEGORICAL = ("weight_class",)


@dataclass
class FighterState:
    """Running, strictly-past career state for one fighter."""

    elo: float = ELO_START
    elo_peak: float = ELO_START
    fights: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    no_contests: int = 0
    streak: int = 0  # positive = win streak, negative = losing streak
    best_win_streak: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))
    ko_wins: int = 0
    sub_wins: int = 0
    dec_wins: int = 0
    ko_losses: int = 0
    sub_losses: int = 0
    dec_losses: int = 0
    total_seconds: float = 0.0
    total_rounds: float = 0.0
    title_bouts: int = 0
    opponent_elo_sum: float = 0.0
    opponent_elo_n: int = 0
    recent_opponent_elo: deque = field(default_factory=lambda: deque(maxlen=3))
    elo_history: deque = field(default_factory=lambda: deque(maxlen=4))
    quality_wins: int = 0
    division_counts: dict = field(default_factory=dict)
    last_division: str = ""
    # Cumulative round-by-round totals over previous bouts. `opp_` fields are the
    # opponents' numbers, which is what defensive rates are built from.
    tally: dict = field(default_factory=dict)
    tally_seconds: float = 0.0
    last_date: pd.Timestamp | None = None
    debut_date: pd.Timestamp | None = None

    def in_fight_features(self) -> dict[str, float]:
        """Per-minute and percentage rates from previous bouts, or NaN if none."""
        t = self.tally
        minutes = self.tally_seconds / 60.0
        if not t or minutes <= 0:
            return {name: np.nan for name in IN_FIGHT_FEATURES}

        def ratio(numerator: str, denominator: str) -> float:
            bottom = t.get(denominator, 0.0)
            return t.get(numerator, 0.0) / bottom if bottom > 0 else np.nan

        landed = t.get("sig_landed", 0.0)
        absorbed = t.get("opp_sig_landed", 0.0)
        per15 = 15.0 / minutes
        return {
            "sig_str_landed_pm": landed / minutes,
            "sig_str_absorbed_pm": absorbed / minutes,
            "sig_str_accuracy": ratio("sig_landed", "sig_attempted"),
            # Defence is the share of the opponents' attempts that missed.
            "sig_str_defense": (
                1.0 - ratio("opp_sig_landed", "opp_sig_attempted")
                if t.get("opp_sig_attempted", 0.0) > 0
                else np.nan
            ),
            "strike_differential_pm": (landed - absorbed) / minutes,
            "takedowns_per15": t.get("td_landed", 0.0) * per15,
            "takedown_accuracy": ratio("td_landed", "td_attempted"),
            "takedown_defense": (
                1.0 - ratio("opp_td_landed", "opp_td_attempted")
                if t.get("opp_td_attempted", 0.0) > 0
                else np.nan
            ),
            "sub_attempts_per15": t.get("sub_att", 0.0) * per15,
            "knockdowns_per15": t.get("kd", 0.0) * per15,
            "knockdowns_absorbed_per15": t.get("opp_kd", 0.0) * per15,
            # Control time is the clearest wrestling signal in the whole dataset.
            "control_share": t.get("ctrl", 0.0) / self.tally_seconds,
            "control_differential": (
                t.get("ctrl", 0.0) - t.get("opp_ctrl", 0.0)
            ) / self.tally_seconds,
            "head_strike_share": ratio("head", "sig_landed"),
            "leg_strike_share": ratio("leg", "sig_landed"),
            "ground_strike_share": ratio("ground", "sig_landed"),
            "distance_strike_share": ratio("distance", "sig_landed"),
        }

    def career_features(
        self, fight_date: pd.Timestamp, weight_class: str = ""
    ) -> dict[str, float]:
        """Snapshot of everything known about this fighter before ``fight_date``."""
        n = self.fights
        # Laplace-smoothed rates: a 1-0 fighter is not a 100% winner.
        denom = float(n) if n else 1.0
        recent = list(self.recent)
        layoff = (fight_date - self.last_date).days if self.last_date is not None else np.nan
        tenure = (fight_date - self.debut_date).days if self.debut_date is not None else 0.0
        history = list(self.elo_history)
        # Trend in the rating over the last few bouts: is this fighter arriving
        # on the way up or on the way down?
        momentum = (self.elo - history[0]) if history else 0.0
        opponents = list(self.recent_opponent_elo)
        return {
            "ufc_fights": float(n),
            "ufc_wins": float(self.wins),
            "ufc_losses": float(self.losses),
            "win_rate": (self.wins + 1.0) / (n + 2.0),
            "win_streak": float(max(self.streak, 0)),
            "lose_streak": float(max(-self.streak, 0)),
            "best_win_streak": float(self.best_win_streak),
            "form_last3": float(np.mean(recent[-3:])) if recent else 0.5,
            "form_last5": float(np.mean(recent)) if recent else 0.5,
            "ko_win_rate": self.ko_wins / denom,
            "sub_win_rate": self.sub_wins / denom,
            "dec_win_rate": self.dec_wins / denom,
            "finish_rate": (self.ko_wins + self.sub_wins) / denom,
            "ko_loss_rate": self.ko_losses / denom,
            "sub_loss_rate": self.sub_losses / denom,
            "finished_rate": (self.ko_losses + self.sub_losses) / denom,
            "avg_fight_minutes": (self.total_seconds / 60.0 / denom) if n else np.nan,
            "total_octagon_minutes": self.total_seconds / 60.0,
            "title_bouts": float(self.title_bouts),
            "elo": self.elo,
            "elo_peak": self.elo_peak,
            "elo_off_peak": self.elo - self.elo_peak,
            "opponent_elo_avg": (
                self.opponent_elo_sum / self.opponent_elo_n if self.opponent_elo_n else np.nan
            ),
            "opponent_elo_last3": float(np.mean(opponents)) if opponents else np.nan,
            "elo_momentum": float(momentum),
            # Wins over opponents who were meaningfully better rated at the time.
            "quality_wins": float(self.quality_wins),
            "division_fights": float(self.division_counts.get(weight_class, 0)),
            # Moving up or down a division is a real disruption, and the raw
            # career record does not show it.
            "changed_division": (
                1.0 if (self.last_division and weight_class != self.last_division) else 0.0
            ),
            "days_since_last_fight": float(layoff) if layoff == layoff else np.nan,
            "days_since_debut": float(tenure),
            "is_ufc_debut": 1.0 if n == 0 else 0.0,
        }

    def record_result(
        self,
        *,
        outcome: str,
        method_class: str,
        seconds: float,
        rounds: float,
        title: bool,
        opponent_elo: float,
        date: pd.Timestamp,
        weight_class: str = "",
        own_elo_before: float = ELO_START,
        round_stats: dict | None = None,
    ) -> None:
        """Fold one completed bout into the running state."""
        if self.debut_date is None:
            self.debut_date = date
        self.last_date = date
        if title:
            self.title_bouts += 1

        if outcome == "NC":
            # A no-contest is not a fight on anybody's record.
            self.no_contests += 1
            return

        if round_stats:
            for key, value in round_stats.items():
                if value == value:
                    self.tally[key] = self.tally.get(key, 0.0) + value
            if seconds == seconds:
                # Only bouts that actually contributed statistics count toward
                # the denominator, so a missing scorecard cannot dilute the rates.
                self.tally_seconds += seconds
        self.fights += 1
        self.opponent_elo_sum += opponent_elo
        self.opponent_elo_n += 1
        self.recent_opponent_elo.append(opponent_elo)
        self.elo_history.append(own_elo_before)
        if weight_class:
            self.division_counts[weight_class] = self.division_counts.get(weight_class, 0) + 1
            self.last_division = weight_class
        if seconds == seconds:
            self.total_seconds += seconds
        if rounds == rounds:
            self.total_rounds += rounds

        if outcome == "W":
            self.wins += 1
            if opponent_elo >= own_elo_before + 50.0:
                self.quality_wins += 1
            self.streak = self.streak + 1 if self.streak > 0 else 1
            self.best_win_streak = max(self.best_win_streak, self.streak)
            self.recent.append(1.0)
            if method_class == "ko_tko":
                self.ko_wins += 1
            elif method_class == "submission":
                self.sub_wins += 1
            else:
                self.dec_wins += 1
        elif outcome == "L":
            self.losses += 1
            self.streak = self.streak - 1 if self.streak < 0 else -1
            self.recent.append(0.0)
            if method_class == "ko_tko":
                self.ko_losses += 1
            elif method_class == "submission":
                self.sub_losses += 1
            else:
                self.dec_losses += 1
        else:  # draw
            self.draws += 1
            self.streak = 0
            self.recent.append(0.5)

    def to_json(self) -> dict:
        data = {
            key: (value if not isinstance(value, deque) else list(value))
            for key, value in self.__dict__.items()
        }
        for key in ("last_date", "debut_date"):
            data[key] = None if data[key] is None else str(pd.Timestamp(data[key]).date())
        return data

    @classmethod
    def from_json(cls, data: dict) -> "FighterState":
        payload = dict(data)
        payload["recent"] = deque(payload.get("recent", []), maxlen=5)
        payload["recent_opponent_elo"] = deque(payload.get("recent_opponent_elo", []), maxlen=3)
        payload["elo_history"] = deque(payload.get("elo_history", []), maxlen=4)
        for key in ("last_date", "debut_date"):
            payload[key] = pd.Timestamp(payload[key]) if payload.get(key) else None
        return cls(**payload)


def elo_update(
    elo_a: float, elo_b: float, score_a: float, method_class: str, fights_a: int, fights_b: int
) -> tuple[float, float]:
    """Symmetric Elo step, scaled by finish decisiveness and by inexperience."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    multiplier = ELO_METHOD_MULTIPLIER.get(method_class, 1.0)
    # Ratings for fighters with almost no UFC history should move faster.
    k_a = ELO_K * multiplier * (1.5 if fights_a < 3 else 1.0)
    k_b = ELO_K * multiplier * (1.5 if fights_b < 3 else 1.0)
    delta_a = k_a * (score_a - expected_a)
    delta_b = k_b * ((1.0 - score_a) - (1.0 - expected_a))
    return elo_a + delta_a, elo_b + delta_b


class RankingLookup:
    """As-of-date divisional and pound-for-pound rankings.

    Rankings are published weekly; for a fight we take the most recent
    publication *strictly before* the fight date.
    """

    def __init__(self, rankings: pd.DataFrame):
        self._div: dict[tuple[str, str], tuple[list, list]] = {}
        self._p4p: dict[str, tuple[list, list]] = {}
        div_rows = defaultdict(list)
        p4p_rows = defaultdict(list)
        for row in rankings.itertuples(index=False):
            key_name = row.name_key
            weight_class = str(row.weightclass)
            if "pound-for-pound" in weight_class.lower():
                p4p_rows[key_name].append((row.date, row.rank))
            else:
                div_rows[(key_name, weight_class)].append((row.date, row.rank))
        for key, values in div_rows.items():
            values.sort()
            self._div[key] = ([v[0] for v in values], [v[1] for v in values])
        for key, values in p4p_rows.items():
            values.sort()
            self._p4p[key] = ([v[0] for v in values], [v[1] for v in values])

    @staticmethod
    def _as_of(store: dict, key, date: pd.Timestamp) -> float:
        entry = store.get(key)
        if not entry:
            return np.nan
        dates, ranks = entry
        idx = bisect.bisect_left(dates, date) - 1
        return float(ranks[idx]) if idx >= 0 else np.nan

    def features(self, name_key: str, weight_class: str, date: pd.Timestamp) -> dict[str, float]:
        rank = self._as_of(self._div, (name_key, weight_class), date)
        p4p = self._as_of(self._p4p, name_key, date)
        ranked = rank == rank
        return {
            # 1.0 = champion, ~0.06 = #15, 0.0 = unranked. Monotone and bounded,
            # which behaves far better in a network than a raw rank integer.
            "div_rank_score": (max(0.0, 16.0 - rank) / 16.0) if ranked else 0.0,
            "is_champion": 1.0 if (ranked and rank == 0) else 0.0,
            "is_ranked": 1.0 if ranked else 0.0,
            "p4p_rank_score": (max(0.0, 16.0 - p4p) / 16.0) if p4p == p4p else 0.0,
        }


def _load_static_attributes(raw_dir: Path) -> dict[str, dict[str, float]]:
    """Height / reach / weight / stance / DOB, which do not change over a career."""
    stats = pd.read_csv(raw_dir / "ufcstats_data.csv")
    stats["name_key"] = stats["name"].map(normalize_name)
    attributes: dict[str, dict] = {}
    for row in stats.itertuples(index=False):
        key = row.name_key
        if not key or key in attributes:
            continue
        attributes[key] = {
            "height_in": parse_height_inches(row.height),
            "reach_in": parse_inches(row.reach),
            "listed_weight": parse_weight_lbs(row.weight),
            "stance": (str(row.stance) if str(row.stance) not in ("nan", "NA", "") else "Unknown"),
            "dob": pd.to_datetime(row.dob, errors="coerce", format="mixed"),
        }
    return attributes


def _static_features(attributes: dict, fight_date: pd.Timestamp) -> dict[str, float]:
    height = attributes.get("height_in", np.nan)
    reach = attributes.get("reach_in", np.nan)
    dob = attributes.get("dob")
    age = np.nan
    if dob is not None and pd.notna(dob):
        age = (fight_date - dob).days / 365.25
    return {
        "age": age,
        # Distance from ~29, roughly the peak of an MMA career. Both the 22-year-old
        # and the 38-year-old are off-peak, which a single monotone age term cannot say.
        "age_from_prime": abs(age - 29.0) if age == age else np.nan,
        "height_in": height,
        "reach_in": reach,
        # "Ape index": reach minus height. A positive value is the classic
        # long-limbed striker's build.
        "ape_index": (reach - height) if (reach == reach and height == height) else np.nan,
        "listed_weight": attributes.get("listed_weight", np.nan),
    }


def _load_market_and_rolling(raw_dir: Path) -> dict:
    """Index the Ultimate UFC dataset by (date, name pair).

    Two things are pulled out of it: the closing moneyline (used only as a
    benchmark, never as a model input unless explicitly asked for) and the
    fighters' pre-fight rolling in-fight statistics.
    """
    wanted = ["r_fighter", "b_fighter", "r_odds", "b_odds", "date"] + [
        f"{corner}_{name}" for corner in ("r", "b") for name in ROLLING_FEATURES
    ]
    ultimate = pd.read_csv(raw_dir / "ultimate_ufc_dataset.csv", usecols=wanted, low_memory=False)
    # Assigned via a dict so pandas does not fragment this very wide frame.
    ultimate = ultimate.assign(
        date=pd.to_datetime(ultimate["date"], errors="coerce"),
        r_key=ultimate["r_fighter"].map(normalize_name),
        b_key=ultimate["b_fighter"].map(normalize_name),
    )

    index: dict[tuple, dict] = {}
    for row in ultimate.itertuples(index=False):
        if pd.isna(row.date) or not row.r_key or not row.b_key:
            continue
        key = (row.date, frozenset((row.r_key, row.b_key)))
        index[key] = {
            "odds": {row.r_key: row.r_odds, row.b_key: row.b_odds},
            "rolling": {
                row.r_key: {
                    "avg_sig_str_landed": row.r_avg_sig_str_landed,
                    "avg_sig_str_pct": row.r_avg_sig_str_pct,
                    "avg_sub_att": row.r_avg_sub_att,
                    "avg_td_landed": row.r_avg_td_landed,
                    "avg_td_pct": row.r_avg_td_pct,
                },
                row.b_key: {
                    "avg_sig_str_landed": row.b_avg_sig_str_landed,
                    "avg_sig_str_pct": row.b_avg_sig_str_pct,
                    "avg_sub_att": row.b_avg_sub_att,
                    "avg_td_landed": row.b_avg_td_landed,
                    "avg_td_pct": row.b_avg_td_pct,
                },
            },
            "red_key": row.r_key,
        }
    return index


def _load_round_stats(raw_dir: Path) -> dict[tuple, dict[str, dict[str, float]]]:
    """Sum UFCStats round rows into per-fight, per-fighter totals.

    Keyed by (normalised event name, frozenset of the two fighter keys), which
    is unique even for the 165 pairs who fought each other more than once.
    """
    path = raw_dir / "ufc_fight_stats.csv"
    if not path.exists():
        return {}
    rounds = pd.read_csv(path).rename(
        columns={
            "SIG.STR.": "sig",
            "TOTAL STR.": "total_str",
            "SUB.ATT": "sub_att",
            "REV.": "rev",
        }
    )

    totals: dict[tuple, dict[str, dict[str, float]]] = {}
    for row in rounds.itertuples(index=False):
        names = [part.strip() for part in str(row.BOUT).split(" vs. ")]
        if len(names) != 2:
            continue
        keys = [normalize_name(name) for name in names]
        fighter = normalize_name(row.FIGHTER)
        if fighter not in keys:
            continue
        bout_key = (normalize_name(row.EVENT), frozenset(keys))
        bucket = totals.setdefault(bout_key, {})
        side = bucket.setdefault(fighter, {})

        sig_landed, sig_attempted = parse_of_pair(row.sig)
        td_landed, td_attempted = parse_of_pair(row.TD)
        head_landed, _ = parse_of_pair(row.HEAD)
        body_landed, _ = parse_of_pair(row.BODY)
        leg_landed, _ = parse_of_pair(row.LEG)
        distance_landed, _ = parse_of_pair(row.DISTANCE)
        clinch_landed, _ = parse_of_pair(row.CLINCH)
        ground_landed, _ = parse_of_pair(row.GROUND)
        contributions = {
            "sig_landed": sig_landed,
            "sig_attempted": sig_attempted,
            "td_landed": td_landed,
            "td_attempted": td_attempted,
            "kd": float(row.KD) if pd.notna(row.KD) else np.nan,
            "sub_att": float(row.sub_att) if pd.notna(row.sub_att) else np.nan,
            "ctrl": parse_clock_seconds(row.CTRL),
            "head": head_landed,
            "body": body_landed,
            "leg": leg_landed,
            "distance": distance_landed,
            "clinch": clinch_landed,
            "ground": ground_landed,
        }
        for name, value in contributions.items():
            if value == value:
                side[name] = side.get(name, 0.0) + value

    # Mirror each fighter's totals onto the opponent as `opp_` fields, which is
    # what the defensive rates (strike defence, takedown defence) are built from.
    for bucket in totals.values():
        fighters = list(bucket)
        if len(fighters) != 2:
            continue
        first, second = fighters
        for own, other in ((first, second), (second, first)):
            for name, value in bucket[other].items():
                bucket[own][f"opp_{name}"] = value
    return totals


def _load_bestfightodds(raw_dir: Path) -> dict[str, dict[str, float]]:
    """Optional supplementary closing lines, keyed by UFCStats fight id.

    Written by ``scripts/fetch_bestfightodds.py``. The Kaggle odds file leaves
    about a fifth of bouts unpriced, which is a gap in that dataset rather than
    in reality; where this file supplies a line for a bout the Kaggle file
    missed, it is used to widen the benchmark sample. Absent, everything below
    behaves exactly as before.
    """
    path = raw_dir / "bestfightodds_odds.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    lookup: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.closing_odds) or not str(row.name_key):
            continue
        lookup.setdefault(str(row.fight_id), {})[str(row.name_key)] = float(row.closing_odds)
    return lookup


def _empty_rolling() -> dict[str, float]:
    return {name: np.nan for name in ROLLING_FEATURES}


def _slot_order(fight_id: str) -> bool:
    """Deterministically decide whether fighter 1 lands in slot A or slot B.

    In the raw file the first-listed fighter wins 64% of the time and the Red
    corner wins 58% — both are artefacts of how promoters and statisticians
    order a bout, not information about the fighters. Shuffling the slots with
    a stable hash removes that signal from the stored dataset entirely, so no
    model trained on it can quietly cash in on the ordering.
    """
    digest = 0
    for char in fight_id:
        digest = (digest * 131 + ord(char)) & 0xFFFFFFFF
    return digest % 2 == 0


def build_features(raw_dir: Path = RAW_DIR, era_start: str = MODEL_ERA_START) -> pd.DataFrame:
    """One chronological pass over every UFC bout, emitting pre-fight features."""
    fights = pd.read_csv(raw_dir / "ufc_fights.csv")
    fights["date"] = pd.to_datetime(fights["date"], errors="coerce")
    fights = fights.dropna(subset=["date", "f1_name", "f2_name", "f1_result"])
    fights = fights.sort_values("date", kind="mergesort").reset_index(drop=True)

    fights["f1_key"] = fights["f1_name"].map(normalize_name)
    fights["f2_key"] = fights["f2_name"].map(normalize_name)

    rankings = pd.read_csv(raw_dir / "ufc_rankings_dataset.csv")
    rankings["date"] = pd.to_datetime(rankings["date"], errors="coerce")
    rankings = rankings.dropna(subset=["date", "fighter", "rank"])
    rankings["name_key"] = rankings["fighter"].map(normalize_name)
    ranking_lookup = RankingLookup(rankings)

    attributes = _load_static_attributes(raw_dir)
    market = _load_market_and_rolling(raw_dir)
    round_stats = _load_round_stats(raw_dir)
    supplementary_odds = _load_bestfightodds(raw_dir)

    states: dict[str, FighterState] = defaultdict(FighterState)
    display_names: dict[str, str] = {}
    last_weight_class: dict[str, str] = {}
    era_start_ts = pd.Timestamp(era_start)

    rows: list[dict] = []
    pending: list[tuple] = []

    def flush(updates: list[tuple]) -> None:
        """Apply one event's results only after every bout on it was emitted."""
        for (
            key1,
            key2,
            outcome1,
            method_class,
            seconds,
            rounds,
            title,
            elo1,
            elo2,
            date,
            bout_class,
            bout_stats,
        ) in updates:
            state1, state2 = states[key1], states[key2]
            if outcome1 in ("W", "L", "D"):
                score1 = {"W": 1.0, "L": 0.0, "D": 0.5}[outcome1]
                state1.elo, state2.elo = elo_update(
                    elo1, elo2, score1, method_class, state1.fights, state2.fights
                )
                state1.elo_peak = max(state1.elo_peak, state1.elo)
                state2.elo_peak = max(state2.elo_peak, state2.elo)
            outcome2 = {"W": "L", "L": "W", "D": "D", "NC": "NC"}[outcome1]
            common = dict(
                method_class=method_class,
                seconds=seconds,
                rounds=rounds,
                title=title,
                date=date,
                weight_class=bout_class,
            )
            state1.record_result(
                outcome=outcome1,
                opponent_elo=elo2,
                own_elo_before=elo1,
                round_stats=bout_stats.get(key1),
                **common,
            )
            state2.record_result(
                outcome=outcome2,
                opponent_elo=elo1,
                own_elo_before=elo2,
                round_stats=bout_stats.get(key2),
                **common,
            )

    current_date = None
    for fight in fights.itertuples(index=False):
        if current_date is not None and fight.date != current_date:
            flush(pending)
            pending = []
        current_date = fight.date

        key1, key2 = fight.f1_key, fight.f2_key
        if not key1 or not key2 or key1 == key2:
            continue
        display_names.setdefault(key1, fight.f1_name)
        display_names.setdefault(key2, fight.f2_name)
        display_names[key1] = fight.f1_name
        display_names[key2] = fight.f2_name

        weight_class = clean_weight_class(fight.weight_class)
        title = is_title_bout(fight.weight_class)
        last_weight_class[key1] = weight_class
        last_weight_class[key2] = weight_class

        method_class = classify_method(fight.method)
        seconds = parse_clock_seconds(fight.time)
        rounds = float(fight.round) if pd.notna(fight.round) else np.nan
        if seconds == seconds and rounds == rounds:
            # Elapsed time = full earlier rounds + the clock in the final round.
            seconds = (rounds - 1) * 300.0 + seconds
        outcome1 = str(fight.f1_result).strip().upper()

        state1, state2 = states[key1], states[key2]
        elo1, elo2 = state1.elo, state2.elo

        pending.append(
            (
                key1,
                key2,
                outcome1,
                method_class,
                seconds,
                rounds,
                title,
                elo1,
                elo2,
                fight.date,
                weight_class,
                round_stats.get(
                    (normalize_name(fight.event_name), frozenset((key1, key2))), {}
                ),
            )
        )

        # Only bouts in the modelled era become training rows; earlier ones
        # exist purely to warm up Elo and the career counters above.
        if fight.date < era_start_ts or outcome1 not in ("W", "L"):
            continue

        market_entry = market.get((fight.date, frozenset((key1, key2))))
        odds = market_entry["odds"] if market_entry else {}
        rolling = market_entry["rolling"] if market_entry else {}

        def side(key: str, state: FighterState) -> dict[str, float]:
            payload = state.career_features(fight.date, weight_class)
            payload.update(state.in_fight_features())
            payload.update(_static_features(attributes.get(key, {}), fight.date))
            payload.update(ranking_lookup.features(key, weight_class, fight.date))
            return payload

        side1, side2 = side(key1, state1), side(key2, state2)
        odds1 = odds.get(key1, np.nan)
        odds2 = odds.get(key2, np.nan)
        if supplementary_odds and (pd.isna(odds1) or pd.isna(odds2)):
            fight_id = str(fight.fight_url).rsplit("/", 1)[-1]
            backup = supplementary_odds.get(fight_id, {})
            # Only fill a side the primary source left blank; never overwrite.
            if pd.isna(odds1):
                odds1 = backup.get(key1, np.nan)
            if pd.isna(odds2):
                odds2 = backup.get(key2, np.nan)

        # Slot A / slot B carry no corner meaning by construction.
        one_first = _slot_order(str(fight.fight_url))
        if one_first:
            a_key, b_key, a_side, b_side = key1, key2, side1, side2
            a_odds, b_odds = odds1, odds2
            label = 1.0 if outcome1 == "W" else 0.0
        else:
            a_key, b_key, a_side, b_side = key2, key1, side2, side1
            a_odds, b_odds = odds2, odds1
            label = 0.0 if outcome1 == "W" else 1.0

        record: dict = {
            "fight_url": fight.fight_url,
            "date": fight.date,
            "event_name": fight.event_name,
            "weight_class": weight_class,
            "a_name": display_names[a_key],
            "b_name": display_names[b_key],
            "a_key": a_key,
            "b_key": b_key,
            "label": label,
            "method_class": method_class,
            "scheduled_rounds": parse_scheduled_rounds(fight.time_format),
            "is_title": 1.0 if title else 0.0,
            "is_womens": 1.0 if "women" in weight_class.lower() else 0.0,
            "era_year": fight.date.year + (fight.date.dayofyear - 1) / 365.25,
            "a_odds": a_odds,
            "b_odds": b_odds,
        }
        for name in FIGHTER_FEATURES:
            record[f"a_{name}"] = a_side.get(name, np.nan)
            record[f"b_{name}"] = b_side.get(name, np.nan)
        # Excluded from the model; retained so the leakage audit can rerun.
        a_roll = rolling.get(a_key) or _empty_rolling()
        b_roll = rolling.get(b_key) or _empty_rolling()
        for name in ROLLING_FEATURES:
            record[f"audit_a_{name}"] = a_roll.get(name, np.nan)
            record[f"audit_b_{name}"] = b_roll.get(name, np.nan)
        record["a_stance"] = attributes.get(a_key, {}).get("stance", "Unknown")
        record["b_stance"] = attributes.get(b_key, {}).get("stance", "Unknown")
        rows.append(record)

    flush(pending)

    frame = pd.DataFrame(rows)
    frame = _attach_market_probabilities(frame)

    snapshot = {
        key: {
            "display_name": display_names.get(key, key),
            "last_weight_class": last_weight_class.get(key, "Unknown"),
            "state": state.to_json(),
            "static": {
                "height_in": attributes.get(key, {}).get("height_in", np.nan),
                "reach_in": attributes.get(key, {}).get("reach_in", np.nan),
                "listed_weight": attributes.get(key, {}).get("listed_weight", np.nan),
                "stance": attributes.get(key, {}).get("stance", "Unknown"),
                "dob": (
                    None
                    if pd.isna(attributes.get(key, {}).get("dob", pd.NaT))
                    else str(pd.Timestamp(attributes[key]["dob"]).date())
                ),
            },
        }
        for key, state in states.items()
    }
    frame.attrs["fighter_snapshot"] = snapshot
    return frame


def _attach_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the vig-free closing-line probability for slot A (benchmark only)."""
    if frame.empty:
        return frame
    implied_a = frame["a_odds"].map(american_to_implied)
    implied_b = frame["b_odds"].map(american_to_implied)
    frame["market_prob_a"] = [
        devig_two_way(x, y) for x, y in zip(implied_a, implied_b, strict=True)
    ]
    frame["market_overround"] = implied_a + implied_b - 1.0
    frame["has_market"] = frame["market_prob_a"].notna().astype(float)
    return frame


def save_features(frame: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURES_CSV, index=False)
    snapshot = frame.attrs.get("fighter_snapshot", {})
    FIGHTER_STATE_JSON.write_text(json.dumps(snapshot, indent=1, default=str))


def load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    return frame
