"""Parsers and odds arithmetic."""

import math

import pytest

from mma_predictor.parsing import (
    american_to_decimal,
    american_to_implied,
    classify_method,
    clean_weight_class,
    devig_two_way,
    normalize_name,
    parse_clock_seconds,
    parse_height_inches,
    parse_percent,
    parse_scheduled_rounds,
    probability_to_american,
)


@pytest.mark.parametrize(
    "raw,expected",
    [("5' 11\"", 71.0), ("6' 0\"", 72.0), ("5' 4\"", 64.0)],
)
def test_parse_height(raw, expected):
    assert parse_height_inches(raw) == expected


def test_parse_height_missing():
    assert math.isnan(parse_height_inches("NA"))


@pytest.mark.parametrize(
    "raw,expected", [("3M 33S", 213.0), ("15S", 15.0), ("5M 0S", 300.0), ("4:20", 260.0)]
)
def test_parse_clock(raw, expected):
    assert parse_clock_seconds(raw) == expected


def test_parse_percent():
    assert parse_percent("38%") == pytest.approx(0.38)
    assert math.isnan(parse_percent("NA"))


def test_scheduled_rounds():
    assert parse_scheduled_rounds("5 Rnd (5-5-5-5-5)") == 5.0
    assert parse_scheduled_rounds("3 Rnd (5-5-5)") == 3.0
    assert parse_scheduled_rounds("No Time Limit") == 1.0


def test_normalize_name_folds_accents_and_punctuation():
    assert normalize_name("José Aldo") == normalize_name("Jose Aldo")
    assert normalize_name("Sean O'Malley") == "sean o malley"
    assert normalize_name(None) == ""


def test_clean_weight_class_collapses_bout_titles():
    """The whole point: one division, not one category per bout title."""
    variants = [
        "Heavyweight Bout",
        "UFC Heavyweight Title Bout",
        "Ultimate Fighter 10 Heavyweight Tournament Title Bout",
    ]
    assert {clean_weight_class(v) for v in variants} == {"Heavyweight"}
    assert clean_weight_class("UFC Interim Light Heavyweight Title Bout") == "Light Heavyweight"
    assert clean_weight_class("UFC Women's Flyweight Title Bout") == "Women's Flyweight"


def test_light_heavyweight_not_swallowed_by_heavyweight():
    assert clean_weight_class("Light Heavyweight Bout") == "Light Heavyweight"


def test_classify_method():
    assert classify_method("KO/TKO") == "ko_tko"
    assert classify_method("Submission") == "submission"
    assert classify_method("Decision - Unanimous") == "decision_unanimous"
    assert classify_method("Decision - Split") == "decision_split"


def test_american_odds_round_trip():
    for probability in (0.2, 0.5, 0.65, 0.9):
        american = probability_to_american(probability)
        assert american_to_implied(american) == pytest.approx(probability, abs=1e-9)


def test_decimal_conversion():
    assert american_to_decimal(-150) == pytest.approx(1 + 100 / 150)
    assert american_to_decimal(+130) == pytest.approx(2.30)


def test_devig_removes_the_overround():
    """Two implied probabilities summing above 1 must come back summing to 1."""
    implied_a, implied_b = american_to_implied(-150), american_to_implied(+130)
    assert implied_a + implied_b > 1.0
    fair_a = devig_two_way(implied_a, implied_b)
    fair_b = devig_two_way(implied_b, implied_a)
    assert fair_a + fair_b == pytest.approx(1.0)
