"""Guards against the two mistakes that make a fight model look better than it is."""

import pandas as pd
import pytest

from mma_predictor.config import SplitConfig
from mma_predictor.dataset import FeatureTransform, chronological_split
from mma_predictor.features import FIGHTER_FEATURES, FighterState, load_features

pytestmark = pytest.mark.skipif(
    not __import__("mma_predictor.config", fromlist=["FEATURES_CSV"]).FEATURES_CSV.exists(),
    reason="run scripts/build_dataset.py first",
)


@pytest.fixture(scope="module")
def frame():
    return load_features()


def test_debutants_have_an_empty_career_record(frame):
    """A fighter with no previous UFC bout must carry no UFC history."""
    debut = frame[frame.a_is_ufc_debut == 1]
    assert len(debut) > 100
    for column in ("a_ufc_fights", "a_ufc_wins", "a_ufc_losses", "a_quality_wins"):
        assert (debut[column] == 0).all(), f"{column} is populated for debutants"
    assert (debut.a_elo == 1500.0).all(), "debutants must sit at the starting Elo"


def test_no_fight_appears_in_two_splits(frame):
    train, val, test = chronological_split(frame, SplitConfig())
    ids = [set(part.fight_url) for part in (train, val, test)]
    assert ids[0] & ids[1] == set()
    assert ids[1] & ids[2] == set()
    assert ids[0] & ids[2] == set()


def test_splits_are_strictly_ordered_in_time(frame):
    """Every training fight must precede every test fight."""
    train, val, test = chronological_split(frame, SplitConfig())
    assert train.date.max() < val.date.min()
    assert val.date.max() < test.date.min()


def test_slot_assignment_carries_no_signal(frame):
    """Slot A must win about half the time, or the slots encode the favourite."""
    assert frame.label.mean() == pytest.approx(0.5, abs=0.02)


def test_transform_statistics_come_from_train_only(frame):
    """Fitting on the full frame must give different numbers than fitting on train."""
    train, _, test = chronological_split(frame, SplitConfig())
    on_train = FeatureTransform.fit(train)
    on_everything = FeatureTransform.fit(frame)
    assert on_train.means != on_everything.means


def test_transform_shares_one_scaler_across_both_slots(frame):
    """Slot A and slot B hold the same quantities and must be scaled identically."""
    train, _, _ = chronological_split(frame, SplitConfig())
    transform = FeatureTransform.fit(train)
    swapped = train.copy()
    for name in FIGHTER_FEATURES:
        swapped[f"a_{name}"], swapped[f"b_{name}"] = (
            train[f"b_{name}"].to_numpy(),
            train[f"a_{name}"].to_numpy(),
        )
    swapped["a_stance"], swapped["b_stance"] = train.b_stance.to_numpy(), train.a_stance.to_numpy()
    original = transform.transform(train)
    flipped = transform.transform(swapped)
    assert (original["xa"] == flipped["xb"]).all()
    assert (original["xb"] == flipped["xa"]).all()


def test_fighter_state_survives_a_json_round_trip():
    state = FighterState()
    state.record_result(
        outcome="W",
        method_class="ko_tko",
        seconds=180.0,
        rounds=1.0,
        title=False,
        opponent_elo=1600.0,
        date=pd.Timestamp("2020-01-01"),
        weight_class="Lightweight",
        own_elo_before=1500.0,
    )
    restored = FighterState.from_json(state.to_json())
    reference = pd.Timestamp("2021-01-01")
    assert restored.career_features(reference, "Lightweight") == state.career_features(
        reference, "Lightweight"
    )
    assert restored.quality_wins == 1
