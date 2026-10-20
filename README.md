# Ball possession from automatic tracking

An entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027),
Phase 1. For every frame of the tactical video of Barcelona v Athletic Club it says who has the
ball: home, away, or nobody because the ball is dead.

| | spellF1 | macroF1 |
|---|---|---|
| Organisers' baseline (smooth50), public 5 min | 0.326 | 0.621 |
| This, public 5 min (17 possessions) | 0.650 | 0.916 |
| This, 7 held-out DFL matches (about 3,000 possessions) | 0.610 | 0.914 |

With 17 possessions, one more or one fewer matched moves the public score by about 0.06, so the
DFL row is the one to trust.

## Running it

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-possession.txt
# match data in ./data as described in DATA.md, or set MTAH_DATA_DIR
python -m possession --out evaluation/results/TEAM.csv
python evaluation/validate_submission.py evaluation/results/TEAM.csv
```

About 40 seconds on a laptop. The fitted models are committed, so nothing is trained here, and
the same data gives the same file byte for byte.

## How it works

The [technical report](docs/technical-report.md) has the details, the numbers and what did not
work. Briefly:

1. Features from the tracking alone, without player identities: who is near the ball and where
   it is heading, how fast everyone moves, each team's shape, and the same a second or two
   either side.
2. Gradient boosting decides whether the ball is dead, who has it when it is live, and who
   restarts when it is not. It is trained on the seven public Sportec/DFL Bundesliga matches,
   which have per-frame labels, after degrading their tracking to look like this match's: long
   ball gaps, aerial balls projected onto the grass, players dropping out.
3. The broadcast video is lined up with the tactical one (one offset per half, found
   automatically). Close-ups and replays usually mean the ball is dead.
4. A Viterbi decoder charges for every change of state, so possessions come out as whole
   intervals instead of flickering.

## Layout

```
possession/   the pipeline; python -m possession writes the submission
models/       fitted classifiers, decoder costs, broadcast weights
scripts/      DFL download, training, broadcast fit, full-match sanity checks
tests/        unit tests that need no data, run in CI against the organisers' scorer
docs/         technical report and build-pdf.sh
```

The rest (CHALLENGE.md, DATA.md, EVALUATION.md, RULES.md, `evaluation/`, `starter/`) is the
organisers' and unchanged. Their README is in the
[challenge repo](https://github.com/metrica-sports/more-than-a-hack-2027#readme).

To retrain `models/`:

```bash
pip install -r requirements-dev.txt
python scripts/get_dfl.py          # about 2.6 GB
python scripts/train.py --no-swap --models models/first/noswap
python scripts/train.py --models models/first/swap
python scripts/fit_broadcast.py    # needs the match data
```

## Data and licence

The match data is licensed for the competition only and is not in this repo: `data/` is
gitignored and has never been committed. Training uses the Sportec Open DFL dataset (Bassek, Rein,
Weber and Memmert, Scientific Data 12:195, 2025, CC-BY 4.0), which `scripts/get_dfl.py`
downloads and which is not redistributed here. The code is MIT, like the organisers'.
