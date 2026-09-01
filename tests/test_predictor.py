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


def test_cross_division_matchup_is_still_order_independent(predictor):
    """The default bout division must not depend on which name was typed first."""
    forward = predictor.predict("Alex Pereira", "Tom Aspinall")
    backward = predictor.predict("Tom Aspinall", "Alex Pereira")
    assert forward.weight_class == backward.weight_class
    assert forward.probability_a == pytest.approx(backward.probability_b, abs=1e-6)


def test_live_features_match_the_training_row(predictor):
    """The prediction path must build the same features the model trained on.

    These are two separate code paths over the same definitions: the training
    pass fills a feature table, while predict.py rebuilds a fighter from the
    stored snapshot. They drifted once -- the in-fight rates were added to the
    training pass only, so every live prediction silently median-imputed 17 of
    its 58 inputs. Replaying a known bout and comparing feature by feature is
    the check that catches that.
    """
    import numpy as np
    import pandas as pd

    from mma_predictor.features import FIGHTER_FEATURES, load_features

    frame = load_features()
    bout = frame[frame.fight_url == frame.iloc[len(frame) // 2].fight_url].iloc[0]
    snapshot = predictor._snapshot_for(pd.Timestamp(bout.date))
    for prefix, key in (("a_", bout.a_key), ("b_", bout.b_key)):
        rebuilt = predictor._side(key, pd.Timestamp(bout.date), bout.weight_class, snapshot)
        for name in FIGHTER_FEATURES:
            assert np.isclose(
                float(bout[prefix + name]), float(rebuilt.get(name, np.nan)),
                rtol=1e-6, equal_nan=True,
            ), f"{prefix}{name} differs between training and prediction paths"
