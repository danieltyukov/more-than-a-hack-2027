# Ball possession from automatic tracking

More than a Hack 2027, Phase 1. Technical document.

## Summary

Three gradient-boosted classifiers look at every frame and say how likely the ball is to be dead,
which side has it if it is alive, and which side will restart if it is dead. They are trained on
the seven public Sportec/DFL Bundesliga matches, whose tracking we degrade until it looks like this
match's automatic tracking, because those matches come with per-frame labels and this one comes
with five minutes. A Viterbi decoder over four states turns the per-frame probabilities into whole
possessions, with a fixed price on every change of state. The broadcast recording, lined up with
the tactical video automatically, adds one more piece of evidence: TV directors cut to close-ups
and replays when the ball is dead.

| | spellF1 | macroF1 | possessions |
|---|---|---|---|
| Public window, our submission | 0.600 | 0.891 | 17 (truth 17) |
| Public window, organisers' baseline (smooth50) | 0.326 | 0.621 | 26 |
| Seven held-out DFL matches, mean | 0.577 | 0.907 | 76% of truth |

The DFL row is the one we trust: every match there is scored by models that never saw it, over
about 3,000 possessions. The public window has 17.

## What the metric rewards

spellF1 counts possessions, not frames. Our first decoder flickered, and every flicker was a false
possession that cost precision and usually split a real one. So most of the design is about whole
intervals: the decoder pays for every change of state, and the per-frame models are judged by the
spellF1 the decoder gets out of them on held-out matches, not by their own accuracy.

Two facts about the data shaped the rest.

- The ball is missing on 41% of graded frames, in long gaps. We found almost no gaps between 1
  and 3 s, so the provider seems to bridge short ones; what is left is 133 gaps of 3 s or more.
  Most missing frames are dead ball (the ball is out of shot or in someone's hands), but around a
  fifth of live frames have no ball either, usually in crowded moments.
- There are five minutes of ground truth, 17 possessions. One possession more or less moves
  spellF1 by about 0.06 there. We used the window as a test and fitted only four numbers on it
  (the broadcast weights below).

## Assets: used, tried, dropped

| Asset | What we did with it |
|---|---|
| Tactical ATD | Everything. Positions, speeds and ball, in metres; no identities needed |
| Broadcast ATD and homography `valid` flag | Aligned to the tactical timeline; the shot type is a dead-ball feature |
| Broadcast ball | Tried as a filler for tactical ball gaps (about 5,900 graded frames). Dropped: it added false turnovers, mostly aerial balls that the two cameras put on different spots of the grass |
| Smart Tagging | Tried as a team prior. Dropped as a model input: per code, its phases agreed with the public ground truth on 44% to 100% of live frames, and the long "final third" and "box" phases span turnovers. Kept for full-match sanity checks (`scripts/sanity.py`) |
| Tactical homography | Not needed, the ATD is already on the pitch |
| Videos | Not used. The ball is about 4 px across in the tactical view, and the frames the tracker lost are mostly occlusions that a second detector would lose too |
| Sportec Open DFL dataset (external) | Training data: seven full matches, 25 Hz, with the provider's per-frame ball status and owning team. CC-BY 4.0 |
| Kickoff frames from DATA.md and FAQ.md | The 2,934 graded frames before the second-half kickoff are dead; the second after each kickoff is live |

## Method

### Features

One row per frame (`possession/framefeatures.py`), identity-free, and computed by the same code
for this match and for the DFL matches:

- Around the ball: nearest and second-nearest player of each side, players within 2 m and
  5 m, ball speed, acceleration and turning angle, ball speed relative to the nearest player of
  each side, how far past a line the ball is
- Where the ball is heading: nearest player of each side to where the ball will be in 0.5 s
  and 1 s at its current velocity (the "ball prediction" idea of Link and Hoernig, 2017)
- Visibility: frames since the ball was last seen and until it is seen again, share seen in
  the surrounding second and four seconds
- Movement: mean player speed per side, and its average over the two seconds before and after
- Touch balance: who is closer to the ball, averaged over 0.25, 1 and 3 s before and after
- Control history: frames since and until each side last clearly had the ball at its feet
  (within 1 m, nobody from the other side within 2.5 m, ball slower than 6 m/s)
- Team shape, per side and in its own attacking direction: centroid, spread, depth, last line,
  forward speed and how tightly each side marks the other, plus the two-second trend of the
  centroid, forward speed and marking

The team-shape group carries a lot when the ball is missing. On its own, without ever seeing the
ball, a classifier on those features named the side in possession on 83% of the public window's
live frames.

### Training data

The DFL matches are corrected multi-camera tracking: every player, the ball on every frame. Each
is degraded before training (`possession/degrade.py`), with amounts drawn at random per copy so
the models cannot fit one setting:

- the ball is hidden in gaps of 3 to 30 s: through most stoppages (60-90% of them), and in 8-24%
  of live play, placed where the ball is crowded
- aerial balls are moved to where the camera ray through them meets the grass, as a single wide
  camera would see them
- players drop out for a few seconds at a time, and positions get slowly varying noise
- every other copy has home and away exchanged, so neither side is special and the models cannot
  learn a home-team prior

One thing had to be taken out rather than put in. The DFL files freeze the ball for the whole of a
stoppage, often past a line, and jump it to the restart spot on the restart frame. Our first
models learned exactly that ("a ball that does not move is dead") and, on this match, put
stoppage starts 2 to 3.5 s late. The stoppage ball is now rebuilt: it rolls on from where play
stopped, slowing down, and is carried at walking pace to the restart spot. The gap edges are
jittered around the whistle as well, so their timing gives nothing away.

The degradation was set against this match from aggregate statistics only: the share of missing
ball frames inside and outside predicted stoppages (77% and 19%), the gap length distribution, and,
roughly, the number of tracked players (20.5 per graded frame here).

### Models

`possession/learned.py`: three `HistGradientBoostingClassifier`s from scikit-learn (400 trees, 63
leaves): dead or alive on all frames; home or away on live frames; home or away to restart on dead
frames. Two degraded copies of each of the seven matches, every second frame, about a million rows.

### Broadcast alignment

The broadcast is not synchronised and has cuts and replays, but it records the same match in real
time, so live broadcast pictures sit at a fixed offset from the tactical video, one per half.
`possession/broadcast.py` finds the offsets by sliding the broadcast ball track along the tactical
one (tactical frame = broadcast frame + 4014 in the first half, + 26920 in the second), corrects
drift of a frame or two window by window, and marks a broadcast frame as live when its players sit
within 1.5 m of the tactical ones.

That gives, for every tactical frame, what the TV director is showing. On the public window, 76%
of dead frames show no pitch at all on the broadcast, against 3% of live frames. The share of the
surrounding second with no pitch, and with a pitch view that is not the live picture, enter the
dead-ball log-odds through a logistic regression on the public window
(`scripts/fit_broadcast.py`): the learned model's log-odds plus those two shares, four weights in
all.

### Decoder

`possession/segment.py`: Viterbi over four hidden states: home, away, dead with home to restart,
dead with away to restart. A dead state can only end in its own side's possession, so the player
standing over the ball during a stoppage (the thrower, the corner taker, the goalkeeper at a goal
kick) decides who restarts, rather than whatever happens in the first second after it. Every
change of state costs a fixed log-probability: 6 for a turnover, 8 to enter or leave a stoppage,
1 to change our mind about who restarts. The costs were chosen by spellF1 on held-out DFL matches;
the curve is flat between 4 and 9 for a turnover (0.575 to 0.577).

## Possession rules as implemented

- A possession changes only when the evidence for the other side, summed from the change onwards,
  pays the switch cost. A pass in flight, a loose ball or a ball nobody can see says little either
  way, so possession stays where it was, which is the ground truth's "the last team keeps it".
- A duel puts players of both sides near the ball and the evidence near zero, so a duel alone
  changes nothing.
- A touch that does not stop the ball shows up as a frame or two of opponent evidence, short of the
  switch cost. A touch after which the opponent keeps the ball pays it.
- Short possessions do come out when the evidence is clear: the DFL labels are full of them and
  the ground truth has them too (0.8 s and 1.3 s in the public window). The decoder costs were
  tuned on matches where they count.
- Dead ball has no owner. While it is dead the decoder carries who will restart, and changes its
  mind only at a price.
- The ball is dead from the start of the second half's graded frames to its kickoff.

## Results

### Held-out DFL matches

Leave-one-match-out: each match scored by models trained on the other six, decoder costs chosen
on the mean (`python scripts/train.py --cv-only`).

Mean over the seven matches: spellF1 0.577, macroF1 0.907, with 76% as many possessions predicted
as there are. Per match, at each match's own best costs, spellF1 ranged from 0.502 to 0.625.

### This match

| Version | Public spellF1 | Public macroF1 | Full match: possessions, dead share, stoppages |
|---|---|---|---|
| Organisers' baseline, smooth50 | 0.326 | 0.621 | 594, 32%, 133 |
| Hand-built rules + Viterbi + broadcast shot (`--rules`) | 0.550 | 0.858 | 268, 38%, 121 |
| Learned models + broadcast shot (submission) | 0.600 | 0.891 | 333, 48%, 129 |

The full-match columns are what `scripts/sanity.py` prints. A dead share near half is what a
106-minute graded window with about 55 minutes of ball in play gives. The literature has about
108 stoppages per match (Siegle and Lames, 2012).

## Limitations and what we would do next

- Stoppage edges. On DFL the decoder finds stoppage starts and restarts to within a few frames.
  On this match all eight stoppage edges in the public window came out late, by 0.3 to 4.3 s: the
  ball usually vanishes the moment it goes out, and the players and the TV director react a second
  or two later. A ball detector run on the tactical video right around the stoppage, or the
  referee's whistle from the audio, would pin the start down.
- Very short possessions. Of the 17 public possessions, the 0.8 s and 1.3 s ones are missed.
  With the ball missing in crowded moments there is often nothing to see them by.
- Label conventions. DFL's owning team flips at almost every opponent touch (7.8-9 possessions
  per live minute); the public window runs at 4.2. Barcelona keep the ball for long spells, which
  explains part of it, but some is definition, and the decoder costs were tuned on DFL's.
- Domain gap. The degradation is our model of the automatic tracker, checked only against
  aggregate statistics. With more of this provider's data, or a few labelled minutes from other
  matches, we would fit the gap pattern directly.
- What we did not try: our own ball detection on the videos, sequence models (a temporal
  convolutional network or a transformer over the whole frame sequence), and fine-tuning on the
  public window.

## Reproduction

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-possession.txt
# unpack the match data into ./data as in DATA.md (or set MTAH_DATA_DIR)
python -m possession --out evaluation/results/TEAM.csv
python evaluation/validate_submission.py evaluation/results/TEAM.csv
```

About 40 s on a laptop, from the raw files. The fitted models are in `models/`, nothing is trained
at this point, and the file is the same byte for byte from run to run.

Rebuilding `models/` from scratch:

```bash
pip install -r requirements-dev.txt
python scripts/get_dfl.py          # the seven DFL matches, about 2.6 GB, checksummed
python scripts/train.py            # leave-one-match-out CV, then the final models
python scripts/fit_broadcast.py    # the four broadcast weights, on the public window
```

## References

- Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal
  and event data in elite soccer. Scientific Data 12:195. doi:10.1038/s41597-025-04505-y. Data:
  doi:10.6084/m9.figshare.28196177
- Link, D. and Hoernig, M. (2017). Individual ball possession in soccer. PLOS ONE 12(7): e0179953.
  doi:10.1371/journal.pone.0179953
- Siegle, M. and Lames, M. (2012). Game interruptions in elite soccer. Journal of Sports Sciences
  30(7): 619-624. doi:10.1080/02640414.2012.667877
- Vidal-Codina, F., Evans, N., El Fakir, B. and Billingham, J. (2022). Automatic event detection in
  football using tracking data. Sports Engineering 25:18. doi:10.1007/s12283-022-00381-6
