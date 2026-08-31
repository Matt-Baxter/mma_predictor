"""The model's defining guarantee: the answer cannot depend on name order."""

import torch

from mma_predictor.model import SymmetricFightNet


def _batch(n=48, n_features=41, n_context=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "xa": torch.randn(n, n_features, generator=generator),
        "xb": torch.randn(n, n_features, generator=generator),
        "sa": torch.randint(0, 5, (n,), generator=generator),
        "sb": torch.randint(0, 5, (n,), generator=generator),
        "xc": torch.randn(n, n_context, generator=generator),
        "wc": torch.randint(0, 13, (n,), generator=generator),
    }


def _model(seed=0):
    torch.manual_seed(seed)
    return SymmetricFightNet(
        n_fighter_features=41, n_context_features=3, n_weight_classes=13, n_stances=5
    ).eval()


def test_probabilities_of_both_orderings_sum_to_one():
    model, batch = _model(), _batch()
    with torch.no_grad():
        forward = torch.sigmoid(
            model(batch["xa"], batch["xb"], batch["sa"], batch["sb"], batch["xc"], batch["wc"])
        )
        reversed_ = torch.sigmoid(
            model(batch["xb"], batch["xa"], batch["sb"], batch["sa"], batch["xc"], batch["wc"])
        )
    assert torch.allclose(forward + reversed_, torch.ones_like(forward), atol=1e-6)


def test_identical_fighters_give_exactly_even_odds():
    model, batch = _model(), _batch()
    with torch.no_grad():
        probability = torch.sigmoid(
            model(batch["xa"], batch["xa"], batch["sa"], batch["sa"], batch["xc"], batch["wc"])
        )
    assert torch.allclose(probability, torch.full_like(probability, 0.5), atol=1e-6)


def test_symmetry_holds_for_several_random_initialisations():
    """It is an architectural property, so it cannot depend on the weights."""
    for seed in range(5):
        model, batch = _model(seed), _batch(seed=seed + 1)
        with torch.no_grad():
            forward = torch.sigmoid(
                model(batch["xa"], batch["xb"], batch["sa"], batch["sb"], batch["xc"], batch["wc"])
            )
            reversed_ = torch.sigmoid(
                model(batch["xb"], batch["xa"], batch["sb"], batch["sa"], batch["xc"], batch["wc"])
            )
        assert torch.allclose(forward + reversed_, torch.ones_like(forward), atol=1e-6)


def test_no_batchnorm_anywhere():
    """BatchNorm would couple the two orderings through the rest of the batch."""
    model = _model()
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
