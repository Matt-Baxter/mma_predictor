"""End-to-end checks on the user-facing prediction path."""

import pytest

from mma_predictor.config import ARTIFACT_DIR, FIGHTER_STATE_JSON
from mma_predictor.predict import FighterNotFound, Predictor

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.pt").exists() or not FIGHTER_STATE_JSON.exists(),
    reason="run scripts/build_dataset.py and python -m mma_predictor.train first",
)


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_swapping_the_argument_order_mirrors_the_answer(predictor):
    """The property that matters to a user typing two names in either order."""
    forward = predictor.predict("Jon Jones", "Stipe Miocic", as_of="2024-11-16")
    backward = predictor.predict("Stipe Miocic", "Jon Jones", as_of="2024-11-16")
    assert forward.probability_a == pytest.approx(backward.probability_b, abs=1e-6)
    assert forward.probability_a + backward.probability_a == pytest.approx(1.0, abs=1e-6)


def test_probabilities_are_a_valid_distribution(predictor):
    matchup = predictor.predict("Islam Makhachev", "Charles Oliveira", as_of="2023-10-21")
    assert 0.0 < matchup.probability_a < 1.0
    assert matchup.probability_a + matchup.probability_b == pytest.approx(1.0)


def test_accents_and_case_resolve_to_the_same_fighter(predictor):
    assert predictor.resolve("jose aldo") == predictor.resolve("José Aldo")


def test_unknown_fighter_raises_with_suggestions(predictor):
    with pytest.raises(FighterNotFound):
        predictor.predict("Zzzz Notafighter", "Jon Jones")


def test_a_fighter_cannot_fight_themselves(predictor):
    with pytest.raises(ValueError):
        predictor.predict("Jon Jones", "Jon Jones")


def test_predictions_move_with_the_as_of_date(predictor):
    """Career state is replayed to a date, so the same pair can price differently."""
    early = predictor.predict("Alexander Volkanovski", "Max Holloway", as_of="2019-12-14")
    late = predictor.predict("Alexander Volkanovski", "Max Holloway", as_of="2026-01-01")
    assert early.probability_a != late.probability_a
