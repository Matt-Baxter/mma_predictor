"""Parsers for the messy string columns in the raw UFC CSVs, plus odds math."""

from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd

_HEIGHT_RE = re.compile(r"""(\d+)\s*'\s*(\d+)?""")
_TIME_RE = re.compile(r"(?:(\d+)\s*M)?\s*(?:(\d+)\s*S)?", re.IGNORECASE)
_ROUNDS_RE = re.compile(r"(\d+)\s*Rnd")


def normalize_name(name: object) -> str:
    """Fold a fighter name to a join key: no accents, no punctuation, lowercase.

    ``Jose Aldo`` and ``José Aldo`` are the same person to us; the raw files
    disagree about which spelling to use.
    """
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_height_inches(value: object) -> float:
    """``5' 11"`` -> 71.0 inches."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    match = _HEIGHT_RE.search(str(value))
    if not match:
        return np.nan
    feet = int(match.group(1))
    inches = int(match.group(2) or 0)
    return float(feet * 12 + inches)


def parse_inches(value: object) -> float:
    """``76"`` -> 76.0 (used for reach)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_weight_lbs(value: object) -> float:
    """``155 lbs.`` -> 155.0."""
    return parse_inches(value)


def parse_percent(value: object) -> float:
    """``38%`` -> 0.38."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(value))
    if not match:
        return np.nan
    number = float(match.group(1))
    return number / 100.0 if "%" in str(value) else number


def parse_clock_seconds(value: object) -> float:
    """``3M 33S`` -> 213.0 seconds. Handles ``15S`` and ``5M`` too."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return np.nan
    if ":" in text:  # some sources use MM:SS
        parts = text.split(":")
        try:
            return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return np.nan
    match = _TIME_RE.fullmatch(text.strip())
    if not match or not any(match.groups()):
        return np.nan
    minutes = int(match.group(1) or 0)
    seconds = int(match.group(2) or 0)
    return float(minutes * 60 + seconds)


def parse_scheduled_rounds(time_format: object) -> float:
    """``5 Rnd (5-5-5-5-5)`` -> 5. Legacy no-limit formats come back as 1."""
    if time_format is None or (isinstance(time_format, float) and math.isnan(time_format)):
        return np.nan
    text = str(time_format)
    match = _ROUNDS_RE.search(text)
    if match:
        return float(match.group(1))
    if "No Time Limit" in text:
        return 1.0
    return np.nan


# Canonical UFC divisions, longest-first so that "Light Heavyweight" is matched
# before "Heavyweight" and "Women's Flyweight" before "Flyweight".
CANONICAL_DIVISIONS = (
    "Women's Bantamweight",
    "Women's Featherweight",
    "Women's Flyweight",
    "Women's Strawweight",
    "Light Heavyweight",
    "Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Strawweight",
    "Catch Weight",
    "Open Weight",
)


# Nominal upper limit in pounds, used only to pick a bout division when two
# fighters come from different ones. Catch/Open weight sort last.
DIVISION_WEIGHT = {
    "Women's Strawweight": 115,
    "Women's Flyweight": 125,
    "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Catch Weight": 999,
    "Open Weight": 999,
}


def pick_bout_division(division_a: str, division_b: str) -> str:
    """Choose the division for a hypothetical bout, independent of argument order.

    If the two fighters last fought in the same division, that is the bout. If
    not, the heavier division wins: the bigger fighter is the one who cannot
    safely make the lighter limit. Deriving it from whichever name was typed
    first would make the prediction depend on argument order, which would break
    the model's symmetry guarantee.
    """
    if division_a == division_b:
        return division_a
    weight_a = DIVISION_WEIGHT.get(division_a, 0)
    weight_b = DIVISION_WEIGHT.get(division_b, 0)
    if weight_a == weight_b:
        return min(division_a, division_b)  # stable tie-break
    return division_a if weight_a > weight_b else division_b


def clean_weight_class(value: object) -> str:
    """Reduce any bout label to its underlying division.

    The raw column is a free-text bout title, so a single division shows up as
    ``Heavyweight Bout``, ``UFC Heavyweight Title Bout``, ``UFC Interim
    Heavyweight Title Bout``, ``Ultimate Fighter 10 Heavyweight Tournament
    Title Bout`` and more. Stripping suffixes piecemeal leaves a hundred
    near-duplicate categories; matching the division name itself leaves
    thirteen, which is the number of divisions the UFC actually has.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Unknown"
    text = str(value)
    lowered = text.lower()
    for division in CANONICAL_DIVISIONS:
        if division.lower() in lowered:
            return division
    return "Unknown"


def is_title_bout(weight_class: object) -> bool:
    return "title" in str(weight_class).lower()


def classify_method(method: object) -> str:
    """Collapse the raw ``method`` column into the buckets Elo and features use."""
    text = str(method or "").lower()
    if "ko/tko" in text or "knockout" in text:
        return "ko_tko"
    if "doctor" in text:
        return "ko_tko"
    if "submission" in text:
        return "submission"
    if "unanimous" in text:
        return "decision_unanimous"
    if "split" in text:
        return "decision_split"
    if "majority" in text:
        return "decision_majority"
    return "other"


def american_to_decimal(odds: float) -> float:
    """-150 -> 1.667, +130 -> 2.30."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return np.nan
    odds = float(odds)
    if odds == 0:
        return np.nan
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def american_to_implied(odds: float) -> float:
    """American moneyline -> raw implied probability (still contains the vig)."""
    decimal = american_to_decimal(odds)
    if decimal is None or (isinstance(decimal, float) and math.isnan(decimal)):
        return np.nan
    return 1.0 / decimal


def probability_to_american(probability: float) -> float:
    """Fair (no-vig) probability -> American moneyline."""
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    if probability >= 0.5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def devig_two_way(implied_a: float, implied_b: float) -> float:
    """Strip the bookmaker's margin from a two-way market, proportionally.

    Returns the fair probability for side A. The two raw implied
    probabilities sum to more than 1 (that excess is the vig / overround);
    normalising by the sum is the standard multiplicative de-vig.
    """
    if any(pd.isna(x) for x in (implied_a, implied_b)):
        return np.nan
    total = implied_a + implied_b
    if total <= 0:
        return np.nan
    return implied_a / total
