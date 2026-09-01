# UFC Fight Predictor

A neural network that takes two UFC fighters' names and returns its own
probability that each one wins — built so that it is *not* a repackaging of the
bookmakers' line.

### ▶ [Try it live](https://matt-baxter.github.io/mma_predictor/) — two search boxes, no install

```
$ python -m mma_predictor.predict "Jon Jones" "Tom Aspinall"

  Jon Jones      63.4%     -173  #########################
  Tom Aspinall   36.6%     +173  ###############
```

## Results

The dataset is 8,319 UFC bouts. Splits are chronological, so the model is always
tested on fights that happened after everything it learned from:

| Split | Period | Bouts | With a closing line |
|---|---|---:|---:|
| Train | 2001 – 2020 | 5,527 | 4,222 |
| Validation | 2021 – May 2023 | 1,205 | 1,137 |
| **Test** | **Jun 2023 – Jun 2026** | **1,587** | **1,214** |
| Total | | 8,319 | 6,573 |

Closing lines exist from 2010 onward and for about 79% of bouts. Everything
below is scored on the test split alone.

| Model | Accuracy | Log loss |
|---|---:|---:|
| Coin flip | 0.505 | 0.6931 |
| Elo rating only | 0.569 | 0.6788 |
| Logistic regression, same features | 0.624 | 0.6415 |
| **This model** | **0.628** | **0.6381** |
| *Betting market (closing line)* | *0.704* | *0.5816* |

Accuracy is how often the favourite it picks actually wins. **Log loss is the
metric that matters**: it scores the probability rather than the pick, so calling
a fight at 90% and being wrong costs far more than being wrong at 51%. Lower is
better, and 0.6931 is what you get by saying 50/50 every time.

Three things are true at once, and the project is only worth reading if all
three are said plainly:

- **It works.** Far better than chance, clearly better than an Elo rating, and
  calibrated to within 2.9% — when it says 65%, that fighter wins 66% of the time.
- **It beats a linear model on the same features, but not by much**, and that is
  a fact about the data rather than a failure of the network — see below.
- **It does not beat the market, and does not pretend to.** The closing line is
  better by 0.06 log loss. Bookmakers know about the injury, the bad weight cut,
  and the fighter who took the bout on nine days' notice. None of that is in any
  public dataset.

## Should the bookmakers' odds be an input? No — with proof

The same architecture, trained twice, differing only in whether it also receives
the vig-free closing line. All three rows are scored on the **1,214 test bouts
that have a closing line**, so they are directly comparable:

| | Accuracy | Log loss |
|---|---:|---:|
| The closing line alone, no model at all | 0.7035 | 0.5816 |
| Network **with** the closing line as an input | 0.7051 | 0.5821 |
| Network **without** it (the shipped model) | 0.6450 | 0.6278 |

The shipped model scores 0.6450 here against 0.628 in the Results table above.
It is the same model; only the set of fights differs. On the 1,214 bouts a
bookmaker priced it gets 0.6450, and on the other 373 it gets close to a coin flip — a
useful finding in itself. Those unpriced bouts are undercard fights between
unknowns, 21% of them involving a UFC debutant against 17% elsewhere, and they
are hard for precisely the reason nobody posts a line on them: there is almost
no career history to work from. **0.628 is the blended figure and the one to
quote.**

Adding the odds moves log loss from 0.6278 to 0.5821. But the odds *by
themselves*, with no model at all, are already worth 0.5816 — so the
network-with-odds ends up a fraction **worse** than simply printing the line.
**It learned to copy the market and contributed nothing of its own**, which is
exactly the failure worth avoiding: it tops the table while having no independent
opinion at all. It looks like the best model in the table while having almost no
independent opinion.

So the odds are used only as the benchmark, never as an input. Reproduce the
variant with `python -m mma_predictor.train --use-market-odds`; it saves to a
separate checkpoint, and `predict.py` refuses to load it, because a model that
needs a bookmaker's line cannot price a hypothetical fight.

**It also finds no betting edge.** Backing every favourite returns +0.5%; betting
at random loses roughly the vig. Betting the model's disagreements with the line
returns −12%, because 80% of them are on the underdog — the model is simply less
confident than the market, and the market is right. This is a learning project,
not a wagering tool.

## Why isn't the network far ahead of logistic regression?

The obvious worry is that the deep model is doing nothing. `scripts/model_comparison.py`
settles it:

| Model | Validation | Test |
|---|---:|---:|
| Logistic regression | 0.6552 | 0.6415 |
| Gradient boosting (best of three settings) | 0.6596 | 0.6482 |
| Logistic regression + squared terms | 0.6557 | 0.6500 |
| **Neural network** | **0.6480** | **0.6381** |

Gradient boosting is excellent at discovering interactions, and it *loses* to the
plain linear model — as do explicit quadratic terms. **The nonlinearity is not
there to find.** Fight outcomes are close to a linear function of these features:
being younger helps, being on a win streak helps, and the effects mostly add up
rather than interacting.

That also explains the fix. The network now carries a **linear skip connection** —
`w · (xₐ − x_b)` added straight to the output, which *is* the antisymmetric
logistic regression. Without it the network had to rediscover that linear
solution through a nonlinear encoder from a random start, and there was no reason
it would land somewhere at least as good; it finished marginally behind. With it,
the linear model sits inside the hypothesis space and the deep branch only learns
the correction. Ablate it with `--no-linear-skip` and test log loss goes back from
0.6381 to 0.6423.

Even so, be honest about the size of the win: a paired t-test over the 1,587 test
bouts gives **p = 0.14**, and the bootstrap 95% CI for the gap is
**[−0.008, +0.001]**. The network leads on both splits, but not by enough to call
it decisively on this much data.

**The ceiling here is the features, not the architecture**, and the clearest
evidence is what happened when the features improved. Adding round-by-round fight
statistics (below) moved test log loss from 0.6426 to 0.6381 — **a bigger gain
than every architecture and hyperparameter change in this project combined.**
Width, depth, dropout, learning rate and batch size were all swept, and the whole
grid moved validation log loss by under 0.005. Better inputs beat better models
here, every time.

## Leakage: two traps in this dataset

Most of the work here was not the network. It was making sure the inputs contain
only what was knowable before the opening horn.

**1. The "pre-fight" rolling averages are not entirely pre-fight.** The source
data ships columns like `r_avg_sig_str_landed`, described as averages over a
fighter's previous bouts. Run `scripts/audit_leakage.py`:

```
Bouts where BOTH fighters are making their UFC debut: 451
(neither man has a previous UFC fight, so every column below must be empty,
 and 0.50 would mean the column says nothing about who won)

column                      populated  predicts winner
avg_sig_str_landed                 70            0.547
avg_sig_str_pct                    81            0.545
```

A column that must be empty is populated in 70 bouts and predicts the winner.
**Excluded** — and then
rebuilt honestly: the model now carries its own strike and takedown rates,
accumulated from raw round-by-round scorecards over each fighter's *previous*
bouts only. Same quantities, with the cutoff enforced here rather than trusted.
Every feature the model sees is computed in this repository from raw fight data,
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

58 features per fighter, all computed in one chronological pass:

- **Career**: record, streaks, recent form, win-method and durability profiles,
  layoff, division changes, and a custom Elo where finishes move the rating more
  than split decisions, plus peak, momentum and strength of schedule.
- **In-fight rates**, accumulated from round-by-round UFCStats data over each
  fighter's *previous* bouts: significant strikes landed and absorbed per minute,
  striking accuracy and defence, takedowns per 15 minutes, takedown accuracy and
  defence, knockdowns, submission attempts, **octagon control share**, and where
  the strikes land (head/leg, distance/ground).
- **Static and contextual**: height, reach, ape index, age and distance from the
  athletic prime, plus as-of-date divisional and pound-for-pound rankings.

Permutation importance on the test set:

```
age                    +0.02100 log loss when shuffled
days_since_debut       +0.00600
sig_str_absorbed_pm    +0.00519
win_streak             +0.00436
takedowns_per15        +0.00391
```

Age dominates everything. After it, five of the next ten most useful features are
the round-by-round ones — and the most useful of those is **strikes absorbed per
minute**, not strikes landed. Taking damage predicts losing better than dealing
it predicts winning.

## Running it

```bash
pip install -r requirements.txt
python scripts/build_dataset.py    # download raw CSVs, build features (~10s)
python -m mma_predictor.train      # 7-model ensemble (~3 min, CPU)
python -m mma_predictor.evaluate   # full report -> artifacts/

python -m mma_predictor.predict "Israel Adesanya" "Sean Strickland"
python -m mma_predictor.predict "Alex Pereira" "Magomed Ankalaev" --title
python -m mma_predictor.predict --search "silva"      # find exact spellings
pytest                                                 # 37 tests
```

`scripts/model_comparison.py` reproduces the linear-vs-nonlinear analysis above.
`scripts/backtest.py` replays held-out fights with the closing line and the real
result beside the model's call (`--fighter`, `--event`, `--disagreements`).

The web app runs the whole ensemble client-side in JavaScript;
`python scripts/export_web.py` refreshes its weights after retraining, and a
push deploys it. `scripts/verify_web.js` runs the page's real code under a stub
DOM and checks it against PyTorch — 28/28 matchups agree to within 1e-5.

## Model and training

A shared encoder embeds each fighter from their features, stance and the bout
context; a pair head reads both encodings, their difference and the context, and
is antisymmetrised as above, plus the linear skip described earlier. ~11k
parameters, trained with AdamW, early stopping on validation log loss, and a
7-seed ensemble. `BatchNorm` is deliberately absent: batch statistics would
couple the two fighter orderings and destroy the symmetry guarantee, so
`LayerNorm` is the only normalisation available here.

Bouts from 1994–2000 sit outside the table above: they only warm up Elo and the
career counters, and never become training rows. One-night open-weight
tournaments under different rules are a different sport.

## Limitations

- **No context the market has**: injuries, short-notice replacements, weight cuts,
  camp changes. This is most of the remaining gap to the closing line.
- **Closing lines cover 79% of bouts**, not because those fights were unpriced but
  because the public odds datasets are incomplete — see below. Odds are only ever
  a benchmark here, so this limits evaluation, not training.
- **Debutants are guesses.** ~24% of bouts involve a fighter with no UFC history.
- **Rankings only start in 2013.**
- `predict.py` uses each fighter's latest career state, so two fighters who
  peaked a decade apart are priced as if both fought today. Use `--as-of` for
  historical questions.

## Data

Two sources, both fetched by `scripts/build_dataset.py`:

1. [TidyTuesday 2026-07-07, "UFC Athletes and Fight Data"](https://github.com/rfordatascience/tidytuesday/tree/main/data/2026/2026-07-07),
   curated by Benjamin Smith from the [`fightr`](https://github.com/benyamindsmith/fightr)
   R package — bout results, fighter profiles, rankings, and the betting odds
   used as a benchmark.
2. [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) —
   round-by-round statistics scraped from [UFCStats](http://ufcstats.com/) and
   republished daily as CSVs. Joins to 99.9% of bouts on (event, fighter pair).

**On the odds gaps.** 21% of bouts have no closing line, which is a limitation of
the public datasets rather than of reality — essentially every UFC fight is
priced. The [Ultimate UFC dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)
used here has blank odds for 16% of 2023 and 25% of 2024, and stops three months
before the fight results do. Its upstream repository
([shortlikeafox](https://github.com/shortlikeafox/ultimate_ufc_dataset)) is
identical and [jansen88/ufc-data](https://github.com/jansen88/ufc-data) covers
less, so there is no more complete public file to switch to;
[BestFightOdds](https://www.bestfightodds.com/) has every fight but would have to
be scraped. Since odds are only ever a benchmark here, the gap limits the size of
the market comparison and nothing else.

MIT licensed. For research and curiosity, not for wagering.
