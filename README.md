# UFC Fight Predictor

A neural network that takes two UFC fighters' names and returns its own
probability that each one wins — built deliberately so that it is *not* a
repackaging of the bookmakers' line.

```
$ python -m mma_predictor.predict "Jon Jones" "Stipe Miocic"

  Jon Jones vs Stipe Miocic
  Heavyweight, 3 rounds | as of 2026-08-31

  Jon Jones      67.7%     -210  ###########################
  Stipe Miocic   32.3%     +210  #############

  Model favourite: Jon Jones (67.7%)
  Odds shown are fair (no vig) and are the model's own estimate,
  produced without any bookmaker line as an input.
```

---

## The headline result, stated honestly

Test set: **1,587 UFC bouts from June 2023 to June 2026** — fights that occur
strictly after every fight the model was trained or validated on.

| Model | n | Accuracy | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| Coin flip | 1587 | 0.505 | 0.6931 | 0.250 | 0.500 |
| Elo formula only | 1587 | 0.569 | 0.6788 | 0.243 | 0.605 |
| Logistic regression (same features) | 1587 | 0.633 | 0.6437 | 0.226 | 0.679 |
| **Neural network (this project)** | 1587 | **0.625** | **0.6457** | **0.227** | **0.681** |
| *Betting market (closing line)* | 1214 | *0.704* | *0.5816* | *0.198* | *0.764* |

Three things are true at once, and the project is only interesting if all three
are said out loud:

1. **The model works.** It is far better than chance and clearly better than a
   plain Elo rating. It is calibrated to within 3.4% expected calibration error —
   when it says 64%, the fighter wins 69% of the time; when it says 35%, they win
   30%.
2. **The network does not beat a linear model on the same features.** Logistic
   regression gets 0.6437 log loss; the network gets 0.6457. On ~5,500 training
   bouts with mostly monotone features, the signal is close to linear. The
   network earns its place through what it *guarantees* (see below), not through
   raw score.
3. **It does not beat the market, and it does not pretend to.** The closing line
   is better by 0.06 log loss — a large gap. Bookmakers know about the injury,
   the visa problem, the bad weight cut, and the fact that a fighter took the
   bout on nine days' notice. None of that is in any public dataset.

## Should the bookmakers' odds be an input? No — and here is the proof

This was the question the project started from, so it got an experiment rather
than an opinion. The same architecture was trained twice, identically, except
that one variant also receives the vig-free closing line as a feature:

| Variant | Accuracy | Log loss | AUC |
|---|---:|---:|---:|
| Closing line alone, no model at all | 0.7035 | **0.5816** | 0.7640 |
| Network **with** the closing line as an input | 0.7076 | **0.5816** | 0.7679 |
| Network **without** it (the shipped model) | 0.6425 | 0.6365 | 0.7013 |

*(All three rows are scored on the 1,214 test bouts that have a closing line.)*

Feeding in the odds gets a log loss of 0.5816. Simply *printing the odds* gets a
log loss of 0.5816. **The market-aware model learned to copy the line and
contributed nothing of its own** — exactly the failure mode worth avoiding. It
looks like the best model in the table and it has no independent opinion at all.

So the odds are used here for three things, and never as an input:

- as the **benchmark** the model is measured against,
- to compute **fair, vig-free probabilities** for that comparison,
- to run a **betting test**, which is the only test that asks whether the model
  knows something the market does not.

You can reproduce the market-aware variant with
`python -m mma_predictor.train --use-market-odds`; it saves to a separate
checkpoint and `predict.py` refuses to load it, because a model that needs a
bookmaker's line cannot price a hypothetical fight.

## Does it find value against the market? No.

| Strategy | Bets | Win rate | ROI |
|---|---:|---:|---:|
| *control:* always back the favourite | 1214 | 0.703 | +0.48% |
| *control:* always back the underdog | 1214 | 0.300 | −15.22% |
| *control:* bet a random side | 1214 | 0.501 | −6.99% |
| Model, bet when it disagrees by >5% | 899 | 0.294 | −16.95% |

The controls are what make this readable. Betting at random loses roughly the
vig, as it must. The model's "value bets" perform like blindly backing
underdogs — and that is precisely what they are: **89% of its edge>5% bets are on
the market's underdog.** The model is systematically less confident than the
market, so every disagreement points the same direction, and the market is right.

**This project is not a betting system and should not be used as one.**

## What makes the model symmetric — and why that matters

A fight has no natural "first fighter", but the raw data is full of ordering
bias: the first-listed fighter in UFCStats wins **64%** of bouts, and the Red
corner wins **58%**, because promoters and statisticians put the favourite there.

A conventional network fed `[fighter_a, fighter_b]` learns that bias, scores well
in validation, and then breaks the moment a user types two names in an arbitrary
order — `P(Jones beats Miocic)` and `P(Miocic beats Jones)` would not even sum to 1.

This model computes a pairwise score and antisymmetrises it:

```
logit(a beats b) = s(a, b) − s(b, a)
```

Swapping the fighters negates the logit, so `P(a beats b) = 1 − P(b beats a)`
holds **exactly**, for every possible input, as a property of the architecture
rather than something coaxed out of the data by augmentation. A fighter matched
against himself returns exactly 0.500. `LayerNorm` is used instead of
`BatchNorm` for the same reason — batch statistics would couple the two
orderings and break the guarantee. This is enforced in `tests/test_symmetry.py`.

Corner and slot information is then discarded entirely: slots are shuffled by a
stable hash at build time, which is why the stored dataset's win rate in slot A
is 0.497.

## Leakage: two things this dataset will hand you if you let it

Most of the work in this project was not the network. It was making sure the
inputs contain only what was knowable before the opening horn.

**1. The "pre-fight" rolling averages are not entirely pre-fight.** The Ultimate
UFC dataset ships columns like `r_avg_sig_str_landed`, described as averages over
a fighter's previous bouts. They are tempting and they are contaminated. The
audit is in `scripts/audit_leakage.py`:

```
Bouts where BOTH fighters are making their UFC debut: 451
(neither man has a previous UFC fight, so every column below must be empty)

column                      populated   AUC vs outcome
avg_sig_str_landed                 70            0.547
avg_sig_str_pct                    81            0.545
```

A column that must be empty is populated in 70 bouts and predicts the winner at
AUC 0.547. **These are excluded from the model.** Every feature the network sees
is computed in `features.py` from raw bout history, static fighter profiles, or
as-of-date rankings.

**2. Missingness is a trap.** A fighter with no reach listed on UFCStats loses
**91%** of their bouts. That is not a fact about reach; it is a proxy for
"obscure short-notice call-up", and possibly for profiles that were never
completed because the fighter went 0–1 and vanished — which would be information
from *after* the fight. Missing values are therefore imputed from training-set
medians, and **no missingness indicators are fed to the model**. This costs
accuracy on purpose.

Beyond that: splits are strictly chronological, all imputation and scaling
statistics are fitted on the training fold alone and shared across both fighter
slots, and every bout on a card is featurised before any of them updates the
running career state, so two fighters on the same event cannot see each other's
result. `tests/test_leakage.py` enforces all of it.

## Features

41 per fighter, all computed in one chronological pass:

- **Career record** — UFC fights, wins, losses, smoothed win rate, current win
  and loss streaks, best streak, form over the last 3 and 5 bouts.
- **Elo** — a custom rating where finishes move the number more than split
  decisions do, and inexperienced fighters move faster. Plus peak Elo, distance
  below peak, momentum over the last 3 bouts, and average opponent Elo (strength
  of schedule).
- **Method profile** — KO/submission/decision win rates, and the rates at which
  the fighter is *finished*, which is what durability looks like in data.
- **Physical** — height, reach, ape index (reach minus height), listed weight,
  age, and distance from the ~29-year-old athletic prime.
- **Context** — division (embedded), title bout, scheduled rounds, layoff since
  last fight, UFC tenure, fights in this division, and whether the fighter is
  changing division for this bout.
- **Rankings** — divisional and pound-for-pound rank as of the most recent
  publication *before* the fight.

Permutation importance on the test set, which lines up with what fight fans
would guess:

```
age                +0.01372 log loss when shuffled
days_since_debut   +0.00421
win_streak         +0.00292
age_from_prime     +0.00255
win_rate           +0.00161
```

Age dominates everything else by a factor of three. Mileage and recent form
follow. Elo matters less than expected, largely because the record and ranking
features already carry much of the same information.

## Quickstart

```bash
git clone https://github.com/Matt-Baxter/mma_predictor
cd mma_predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_dataset.py        # download raw CSVs, build features (~10s)
python -m mma_predictor.train          # train the 7-model ensemble (~3 min, CPU)
python -m mma_predictor.evaluate       # full report -> artifacts/

python -m mma_predictor.predict "Israel Adesanya" "Sean Strickland"
```

Useful flags:

```bash
python -m mma_predictor.predict "Alex Pereira" "Magomed Ankalaev" --title --rounds 5
python -m mma_predictor.predict "Jon Jones" "Tom Aspinall" --as-of 2025-01-01
python -m mma_predictor.predict --search "silva"      # find exact spellings
python -m mma_predictor.predict "Jon Jones" "Ciryl Gane" --json
pytest                                                 # 34 tests
```

`--as-of` replays each fighter's career state to that date, so you can ask what
the model would have said before a fight that has since happened.

### Backtesting

`scripts/backtest.py` replays real bouts from the held-out test split — fights
the model never trained on — priced from each fighter's career as it stood the
day before. The actual result and the closing line sit next to the model's call:

```bash
python scripts/backtest.py --fighter "Sean O'Malley"
python scripts/backtest.py --event "UFC 300"
python scripts/backtest.py --disagreements   # biggest gaps vs the market
python scripts/backtest.py --upsets          # underdogs it called correctly
```

```
  date       model pick                conf  market  result  actual winner
  2024-04-13 Alex Pereira              61%     54%  HIT     Alex Pereira
  2024-04-13 Max Holloway              51%     38%  HIT     Max Holloway
  2024-04-13 Bo Nickal                 68%     90%  HIT     Bo Nickal
  2024-04-13 Holly Holm                63%     21%  miss    Kayla Harrison
  ...
  model: 9/13 correct (69%)   market on the same bouts: 7/9 (78%)
```

Two warnings about reading these. `--upsets` selects bouts *because* the model
got them right, so its hit rate is 100% by construction and proves nothing; the
script says so when you run it. And any single card is 13 fights — far too few
to separate skill from luck. The number that counts is the full 1,587-bout
evaluation above.

## Layout

```
mma_predictor/
  config.py      paths, Elo constants, split boundaries, hyperparameters
  download.py    fetch the five raw CSVs
  parsing.py     height/clock/percentage parsers, name folding, odds arithmetic
  features.py    the chronological pass -- the core of the project
  dataset.py     chronological splits, train-only imputation and scaling
  model.py       the antisymmetric network
  train.py       training loop, early stopping, ensembling
  evaluate.py    baselines, calibration, betting test, permutation importance
  predict.py     the two-names-in, probabilities-out CLI
scripts/
  build_dataset.py   download + featurise
  audit_leakage.py   reproduce the leakage finding above
  sweep.py           validation-only hyperparameter search
tests/                34 tests
```

## Model and training

A shared encoder embeds each fighter from their 41 features plus a stance
embedding and the shared bout context (division embedding, title flag, scheduled
rounds). A pair head reads both encodings, their difference, and the context, and
is antisymmetrised as described above. About 11k parameters.

Trained with AdamW, cosine-decayed learning rate, dropout 0.35, label smoothing
0.03, early stopping on validation log loss, and a 7-model seed ensemble whose
probabilities are averaged (averaging preserves antisymmetry).

The network is small on purpose. `scripts/sweep.py` searched widths from a single
48-unit layer up to `(128, 96)`, three dropout rates and three learning rates:
**every configuration landed within ~0.005 validation log loss of every other**,
which is inside the noise of a 1,205-fight validation set. Under the
one-standard-error rule the smallest model in that band is the honest choice.

Splits: train 2001–2020 (5,527 bouts), validation 2021 – May 2023 (1,205), test
June 2023 – June 2026 (1,587). Bouts from 1994–2000 are used only to warm up Elo
and career counters — the Unified Rules arrived in late 2000, and one-night
open-weight tournaments are a different sport.

## Honest limitations

- **No fight-level statistics.** Round-by-round strikes, takedowns and control
  time exist on UFCStats but not in these files, and the pre-aggregated versions
  that ship with the dataset failed the leakage audit. This is the single
  biggest thing missing.
- **No context the market has**: injuries, short-notice replacements, weight-cut
  problems, camp changes, visa issues.
- **Debutants are guesses.** Roughly 24% of bouts involve at least one fighter with no UFC
  history, and for them the model has physical attributes and nothing else.
- **Rankings only start in 2013**, so ranking features are zero for earlier bouts.
- **`predict.py` uses each fighter's latest career state**, so a matchup between
  two fighters who peaked a decade apart is priced as if both fought today. Use
  `--as-of` for historical questions.

## Data

[TidyTuesday 2026-07-07, "UFC Athletes and Fight Data"](https://github.com/rfordatascience/tidytuesday/tree/main/data/2026/2026-07-07),
curated by Benjamin Smith from the [`fightr`](https://github.com/benyamindsmith/fightr)
R package, which compiles UFC athlete profiles, [UFCStats](http://ufcstats.com/),
the [Ultimate UFC Kaggle dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)
and the [Octagon API](https://www.octagon-api.com/). Raw files are not committed;
`scripts/build_dataset.py` fetches them.

## Licence

MIT. For research and curiosity, not for wagering.
