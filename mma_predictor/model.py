"""A fight-outcome network that is symmetric by construction.

The central design decision of this project lives in :meth:`SymmetricFightNet.forward`.

A fight has no natural "first fighter". The raw data disagrees: the
first-listed fighter in UFCStats wins 64% of the time and the Red corner wins
58%, purely because matchmakers and statisticians put the favourite there. A
plain MLP fed ``[fighter_a, fighter_b]`` will happily learn that bias, look
excellent in validation, and then fall apart the moment a user types two names
in an arbitrary order -- ``P(Jones beats Miocic)`` and ``P(Miocic beats Jones)``
would not even add to 1.

So the network computes a *pairwise score* and antisymmetrises it:

    logit(a beats b) = s(a, b) - s(b, a)

Swapping the two fighters negates the logit, so ``P(a beats b) = 1 - P(b beats a)``
holds exactly, to floating-point precision, for every possible input. It is a
property of the architecture, not something coaxed out of the data by
augmentation, and it means the model has no corner bias to exploit in the first
place.

``LayerNorm`` is used rather than ``BatchNorm`` for the same reason: batch
statistics would couple the two orderings through the rest of the batch and
break the guarantee. The one place symmetry is not bit-exact is under dropout
during training, where the two calls draw different masks; in ``eval()`` mode,
which is what every prediction and every evaluation uses, it is exact.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(sizes: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SymmetricFightNet(nn.Module):
    def __init__(
        self,
        n_fighter_features: int,
        n_context_features: int,
        n_weight_classes: int,
        n_stances: int,
        hidden_fighter: tuple[int, ...] = (128, 96),
        hidden_pair: tuple[int, ...] = (128, 64),
        embed_dim: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.weight_class_embedding = nn.Embedding(n_weight_classes, embed_dim)
        self.stance_embedding = nn.Embedding(n_stances, embed_dim)

        context_dim = n_context_features + embed_dim
        encoder_in = n_fighter_features + embed_dim + context_dim
        self.fighter_encoder = _mlp((encoder_in, *hidden_fighter), dropout)

        # The head sees both fighters, their difference, and the shared context.
        pair_in = 3 * hidden_fighter[-1] + context_dim
        self.pair_body = _mlp((pair_in, *hidden_pair), dropout)
        self.pair_out = nn.Linear(hidden_pair[-1], 1)

        # Start every fight as a coin flip.
        nn.init.zeros_(self.pair_out.bias)
        nn.init.normal_(self.pair_out.weight, std=0.01)

    def encode_fighter(
        self, features: torch.Tensor, stance: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        stance_vector = self.stance_embedding(stance)
        return self.fighter_encoder(torch.cat([features, stance_vector, context], dim=-1))

    def _score(
        self, encoded_a: torch.Tensor, encoded_b: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.cat([encoded_a, encoded_b, encoded_a - encoded_b, context], dim=-1)
        return self.pair_out(self.pair_body(pair))

    def forward(
        self,
        xa: torch.Tensor,
        xb: torch.Tensor,
        sa: torch.Tensor,
        sb: torch.Tensor,
        xc: torch.Tensor,
        wc: torch.Tensor,
    ) -> torch.Tensor:
        """Return the logit that fighter A beats fighter B."""
        context = torch.cat([xc, self.weight_class_embedding(wc)], dim=-1)
        encoded_a = self.encode_fighter(xa, sa, context)
        encoded_b = self.encode_fighter(xb, sb, context)
        # The antisymmetrisation. This single line is what makes the model's
        # answer independent of the order the two names were given in.
        logit = self._score(encoded_a, encoded_b, context) - self._score(
            encoded_b, encoded_a, context
        )
        return logit.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(
            self(batch["xa"], batch["xb"], batch["sa"], batch["sb"], batch["xc"], batch["wc"])
        )
