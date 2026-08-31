# UFC Fight Predictor

A neural network that takes two UFC fighters' names and returns its own
probability that each one wins — built so that it is *not* a repackaging of the
bookmakers' line.

### ▶ [Try it live](https://matt-baxter.github.io/mma_predictor/) — two search boxes, no install

```
$ python -m mma_predictor.predict "Jon Jones" "Stipe Miocic"

  Jon Jones      67.7%     -210  ###########################
  Stipe Miocic   32.3%     +210  #############
```

## Results

Tested on **1,587 bouts from June 2023 to June 2026** — every one of them
occurring after the last fight the model was trained on.

| Model | Accuracy | Log loss | AUC |
|---|---:|---:|---:|
| Coin flip | 0.505 | 0.6931 | 0.500 |
| Elo rating only | 0.569 | 0.6788 | 0.605 |
| Logistic regression, same features | 0.633 | 0.6437 | 0.679 |
| **This model** | **0.625** | **0.6457** | **0.681** |
| *Betting market (closing line)* | *0.704* | *0.5816* | *0.764* |

Three things are true at once, and the project is only worth reading if all
three are said plainly:

- **It works.** Far better than chance, clearly better than an Elo rating, and
  calibrated to within 3.4% — when it says 64%, that fighter wins 69% of the time.
- **It does not beat a linear model on the same features.** With ~5,500 training
  bouts and mostly monotone inputs, the signal is close to linear. The network
  earns its place through what it *guarantees*, not its score.
- **It does not beat the market, and does not pretend to.** The closing line is
  better by 0.06 log loss. Bookmakers know about the injury, the bad weight cut,
  and the fighter who took the bout on nine days' notice. None of that is in any
  public dataset.

## Should the bookmakers' odds be an input? No — with proof

The same architecture, trained twice, differing only in whether it also receives
the vig-free closing line. Scored on the 1,214 test bouts that have one:

| | Accuracy | Log loss |
|---|---:|---:|
| The closing line alone, no model at all | 0.7035 | **0.5816** |
| Network **with** the closing line as an input | 0.7076 | **0.5816** |
| Network **without** it (the shipped model) | 0.6425 | 0.6365 |

Feeding in the odds scores 0.5816. Simply *printing* the odds scores 0.5816.
**The market-aware model learned to copy the line and contributed nothing of its
own.** It looks like the best model in the table while having no independent
opinion at all.

So the odds are used only as the benchmark, never as an input. Reproduce the
variant with `python -m mma_predictor.train --use-market-odds`; it saves to a
separate checkpoint, and `predict.py` refuses to load it, because a model that
needs a bookmaker's line cannot price a hypothetical fight.

**It also finds no betting edge.** Backing every favourite returns +0.5%; betting
at random loses roughly the vig. Betting the model's disagreements with the line
returns −17%, because 89% of them are on the underdog — the model is simply less
confident than the market, and the market is right. This is a learning project,
not a wagering tool.

## Leakage: two traps in this dataset

Most of the work here was not the network. It was making sure the inputs contain
only what was knowable before the opening horn.

**1. The "pre-fight" rolling averages are not entirely pre-fight.** The source
data ships columns like `r_avg_sig_str_landed`, described as averages over a
fighter's previous bouts. Run `scripts/audit_leakage.py`:

```
Bouts where BOTH fighters are making their UFC debut: 451
(neither man has a previous UFC fight, so every column below must be empty)

column                      populated   AUC vs outcome
avg_sig_str_landed                 70            0.547
avg_sig_str_pct                    81            0.545
```

A column that must be empty is populated in 70 bouts and predicts the winner.
**Excluded.** Every feature the model sees is computed from raw bout history,
static fighter profiles, or as-of-date rankings.

**2. Missingness is a trap.** A fighter with no reach listed loses **91%** of
their bouts. That is not a fact about reach — it is a proxy for "obscure
short-notice call-up", and possibly for profiles never completed because the
fighter went 0–1 and vanished, which would be information from *after* the
fight. Missing values are imputed from training-set medians with **no
missingness indicators**. This costs accuracy on purpose.

Beyond that: splits are strictly chronological, scaling statistics are fitted on
the training fold alone, and every bout on a card is featurised before any of
them updates the running career state. `tests/test_leakage.py` enforces it.

## Symmetric by construction

A fight has no natural "first fighter", but the data is full of ordering bias:
the first-listed fighter wins **64%** of bouts and the Red corner **58%**,
because promoters put the favourite there. A conventional network fed
`[fighter_a, fighter_b]` learns that bias, validates well, then breaks the moment
a user types two names in an arbitrary order.

This model computes a pairwise score and antisymmetrises it:

```
logit(a beats b) = s(a, b) − s(b, a)
```

Swapping the fighters negates the logit, so `P(a beats b) = 1 − P(b beats a)`
holds **exactly**, for every possible input, as a property of the architecture
rather than something coaxed out of the data by augmentation. A fighter against
himself returns exactly 0.500. `LayerNorm` replaces `BatchNorm` for the same
reason — batch statistics would couple the two orderings and break the
guarantee. Enforced in `tests/test_symmetry.py`.

## What the model sees

41 features per fighter, all computed in one chronological pass: career record
and streaks, a custom Elo (finishes move it more than split decisions, plus peak,
momentum and strength of schedule), win-method and durability profiles, physical
attributes, layoff, division changes, and as-of-date rankings.

Permutation importance on the test set:

```
age                +0.01372 log loss when shuffled
days_since_debut   +0.00421
win_streak         +0.00292
age_from_prime     +0.00255
win_rate           +0.00161
```

Age dominates by a factor of three; mileage and recent form follow. Elo matters
less than expected — the record and ranking features already carry much of it.

## Running it

```bash
pip install -r requirements.txt
python scripts/build_dataset.py    # download raw CSVs, build features (~10s)
python -m mma_predictor.train      # 7-model ensemble (~3 min, CPU)
python -m mma_predictor.evaluate   # full report -> artifacts/

python -m mma_predictor.predict "Israel Adesanya" "Sean Strickland"
python -m mma_predictor.predict "Alex Pereira" "Magomed Ankalaev" --title
python -m mma_predictor.predict --search "silva"      # find exact spellings
pytest                                                 # 35 tests
```

`scripts/backtest.py` replays held-out fights with the closing line and the real
result beside the model's call (`--fighter`, `--event`, `--disagreements`).

The web app runs the whole ensemble client-side in JavaScript;
`python scripts/export_web.py` refreshes its weights after retraining, and a
push deploys it. `scripts/verify_web.js` runs the page's real code under a stub
DOM and checks it against PyTorch — 28/28 matchups agree to within 1e-5.

## Model and training

A shared encoder embeds each fighter from their features, stance and the bout
context; a pair head reads both encodings, their difference and the context, and
is antisymmetrised as above. ~11k parameters, trained with AdamW, early stopping
on validation log loss, and a 7-seed ensemble. The network is deliberately small:
a sweep from one 48-unit layer up to `(128, 96)` moved validation log loss by
less than 0.005, which is inside the noise of a 1,205-fight validation set.

Splits: train 2001–2020 (5,527 bouts), validation 2021–May 2023 (1,205), test
June 2023–June 2026 (1,587). Bouts from 1994–2000 only warm up Elo and career
counters — one-night open-weight tournaments are a different sport.

## Limitations

- **No fight-level statistics.** Round-by-round strikes, takedowns and control
  time are not in these files, and the pre-aggregated versions failed the audit
  above. The biggest thing missing.
- **No context the market has**: injuries, short-notice replacements, weight cuts.
- **Debutants are guesses.** ~24% of bouts involve a fighter with no UFC history.
- **Rankings only start in 2013.**
- `predict.py` uses each fighter's latest career state, so two fighters who
  peaked a decade apart are priced as if both fought today. Use `--as-of` for
  historical questions.

## Data

[TidyTuesday 2026-07-07, "UFC Athletes and Fight Data"](https://github.com/rfordatascience/tidytuesday/tree/main/data/2026/2026-07-07),
curated by Benjamin Smith from the [`fightr`](https://github.com/benyamindsmith/fightr)
R package (UFC athlete profiles, [UFCStats](http://ufcstats.com/), the
[Ultimate UFC Kaggle dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset),
and the [Octagon API](https://www.octagon-api.com/)). Raw files are not
committed; `scripts/build_dataset.py` fetches them.

MIT licensed. For research and curiosity, not for wagering.
