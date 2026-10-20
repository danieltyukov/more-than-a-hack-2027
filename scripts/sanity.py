"""Full-match sanity checks for a submission, against things that are not the ground truth.

    python scripts/sanity.py evaluation/results/possession.csv [more.csv ...]

The public ground truth covers five minutes. These numbers cover the whole match, so they catch
things the five minutes cannot: a decoder that invents possessions in the second half, a dead-ball
model that never lets play restart, a restart side that is wrong half the time.

  possessions, mean length, share of graded time dead, number of stoppages
  set pieces   tagged set pieces (Smart Tagging) with a predicted restart inside the tagged
               window, give or take two seconds
  phase team   share of live predicted frames inside a tagged attacking or defending phase whose
               side agrees with the tag. The tags are automatic and their phases span turnovers,
               so 1.0 is not the target; on the public window the tags themselves agree with the
               ground truth 44-100% depending on the code
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402
from possession import tagging  # noqa: E402

SLACK = 50
FPS = 25


def check(path, events):
    codes = pm.label_codes(pm.read_label_csv(path))
    graded = np.full(source_frames(), pm.X, dtype=np.int8)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = pm.H
    runs = pm.valid_runs(graded)
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.possession_spells(filled, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / FPS
    stops = [s for s in pm.segments(filled, runs) if s[2] == pm.D]
    in_play = graded != pm.X

    restarts = np.array([f for f, a, b in pm.transitions(filled, runs) if a == pm.D])
    tagged = [(s, e) for code, s, e, _, _ in events if code == "set pieces" and in_play[s]]
    hit = sum(1 for s, e in tagged
              if len(restarts) and np.any((restarts >= s - SLACK) & (restarts <= e + SLACK)))

    votes = tagging.possession_prior(len(codes), events)
    live = np.isin(filled, (pm.H, pm.A)) & (votes != 0) & in_play
    agree = float(np.mean((filled[live] == pm.H) == (votes[live] > 0)))

    print("{}\n  possessions {}  mean {:.1f} s  under 2 s {}  dead {:.1%}  stoppages {}\n"
          "  set pieces with a restart nearby {}/{}  phase team agreement {:.3f}".format(
              os.path.basename(path), len(spells), lengths.mean(), int((lengths < 2).sum()),
              float(np.mean(filled[in_play] == pm.D)), len(stops), hit, len(tagged), agree))


def main():
    events = tagging.load()
    for path in sys.argv[1:]:
        check(path, events)


if __name__ == "__main__":
    main()
