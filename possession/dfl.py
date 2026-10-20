"""Read a Sportec Open DFL match into the same arrays as the competition tracking, with labels.

The DFL position files carry, for the ball, the provider's per-frame BallStatus (0 dead, 1 alive)
and BallPossession (1 home, 2 away). Turned into H/A/D that is the same kind of label as the
competition ground truth, which is why these matches are the training set here.

Coordinates are converted to the competition convention: metres from the top-left corner, x along
the long axis. The two halves are laid end to end with a gap of NaN frames between them, so the
result looks like a match video and the same per-half code runs on both.
"""

import glob
import hashlib
import os

import numpy as np
from lxml import etree

from .atd import AWAY, CACHE_DIR, HOME, ROOT, Tracking

DFL_DIR = os.path.join(ROOT, "external", "dfl")
HALF_GAP = 500                       # NaN frames between the halves
LABEL_H, LABEL_A, LABEL_D, LABEL_X = 0, 1, 2, 3


def match_ids(folder=DFL_DIR):
    return sorted(os.path.basename(p).split("DFL-MAT-")[1][:-4]
                  for p in glob.glob(os.path.join(folder, "DFL_04_03_positions_raw_observed_*.xml")))


def _files(match_id, folder):
    pos = glob.glob(os.path.join(folder, "DFL_04_03_positions_raw_observed_*{}.xml".format(match_id)))
    info = glob.glob(os.path.join(folder, "DFL_02_01_matchinformation_*{}.xml".format(match_id)))
    return pos[0], info[0]


def _teams(info_path):
    root = etree.parse(info_path).getroot()
    roles = {}
    for team in root.iter("Team"):
        roles[team.get("TeamId")] = HOME if team.get("Role") == "home" else AWAY
    pitch = root.find(".//Environment")
    size = (float(pitch.get("PitchX", 105.0)), float(pitch.get("PitchY", 68.0)))
    return roles, size


def _parse(pos_path, roles):
    """{(section, person): (frames, x, y, s, z, possession, status)} straight from the XML."""
    sets = {}
    section = person = team = None
    rows = []
    for event, el in etree.iterparse(pos_path, events=("start", "end"), tag=("FrameSet", "Frame")):
        if el.tag == "FrameSet":
            if event == "start":
                section, person, team = el.get("GameSection"), el.get("PersonId"), el.get("TeamId")
                rows = []
            else:
                if team == "BALL" or team in roles:
                    sets[(section, person, team)] = np.array(rows, dtype=np.float64)
                el.clear()
            continue
        if event != "end":
            continue
        g = el.get
        if team == "BALL":
            rows.append((float(g("N")), float(g("X")), float(g("Y")), float(g("S")),
                         float(g("Z", "nan")), float(g("BallPossession", "nan")),
                         float(g("BallStatus", "nan"))))
        elif team in roles:
            rows.append((float(g("N")), float(g("X")), float(g("Y")), float(g("S"))))
        el.clear()
    return sets


class LabelledTracking(Tracking):
    """A Tracking with per-frame H/A/D labels and explicit half blocks [(half, first, last)]."""

    def __init__(self, players, speed, ball, ball_z, team, pitch, labels, blocks, name):
        super().__init__(players, speed, ball, ball_z, team, pitch, blocks)
        self.labels = labels
        self.name = name


def load(match_id, folder=DFL_DIR, use_cache=True):
    pos_path, info_path = _files(match_id, folder)
    key = hashlib.sha1("{}:{}".format(pos_path, os.path.getsize(pos_path)).encode()).hexdigest()[:10]
    cache = os.path.join(CACHE_DIR, "dfl_{}_{}.npz".format(match_id, key))
    if use_cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
        return LabelledTracking(z["players"], z["speed"], z["ball"], z["ball_z"], z["team"],
                                tuple(z["pitch"]), z["labels"], blocks, match_id)

    roles, (length, width) = _teams(info_path)
    sets = _parse(pos_path, roles)
    sections = sorted({s for s, _, _ in sets}, key=lambda s: ("first" not in s, s))
    people = sorted({(p, t) for _, p, t in sets if t != "BALL"}, key=lambda pt: (roles[pt[1]], pt[0]))
    column = {p: i for i, (p, _) in enumerate(people)}
    team = np.array([roles[t] for _, t in people], dtype=np.int8)

    spans = {}
    for (s, p, t), arr in sets.items():
        lo, hi = int(arr[:, 0].min()), int(arr[:, 0].max())
        a, b = spans.get(s, (lo, hi))
        spans[s] = (min(a, lo), max(b, hi))
    starts, blocks, n = {}, [], 0
    for h, s in enumerate(sections, start=1):
        if h > 1:
            n += HALF_GAP
        starts[s] = n - spans[s][0]
        length_s = spans[s][1] - spans[s][0] + 1
        blocks.append((h, n, n + length_s - 1))
        n += length_s

    players = np.full((n, len(people), 2), np.nan, dtype=np.float32)
    speed = np.full((n, len(people)), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball_z = np.full(n, np.nan, dtype=np.float32)
    labels = np.full(n, LABEL_X, dtype=np.int8)
    for (s, p, t), arr in sets.items():
        idx = arr[:, 0].astype(np.int64) + starts[s]
        x = arr[:, 1] + length / 2.0
        y = width / 2.0 - arr[:, 2]
        if t == "BALL":
            ball[idx, 0], ball[idx, 1] = x, y
            ball_z[idx] = arr[:, 4]
            alive = arr[:, 6] == 1
            labels[idx] = np.where(~alive, LABEL_D, np.where(arr[:, 5] == 1, LABEL_H, LABEL_A))
        else:
            c = column[p]
            players[idx, c, 0], players[idx, c, 1] = x, y
            speed[idx, c] = arr[:, 3] / 3.6

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players, speed=speed, ball=ball, ball_z=ball_z, team=team,
                 pitch=np.array([length, width]), labels=labels, blocks=np.array(blocks))
    return LabelledTracking(players, speed, ball, ball_z, team, (length, width), labels, blocks,
                            match_id)
