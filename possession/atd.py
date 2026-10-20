"""Read the raw ATD into arrays and cache them.

A raw line is "<frame_id>:<slot>;<slot>;...;<slot>:<ball>" with every group "x,y,s" and NaN for a
missing value. 46 slots, frame_id 1..199319, so row i of every array here is video frame i, which
is also submission row i.

Parsing the text takes ~15 s, so the arrays are written to cache/ next to the repository root and
reused. cache/ is gitignored: it is the tracking data in another format and must not be published.
"""

import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")

HOME, AWAY = 0, 1


class Tracking:
    """Positions in metres, origin at the top-left corner flag, x along the long axis.

    players  (n, k, 2) float32, NaN where the slot is not tracked
    speed    (n, k)    float32, m/s as reported by the tracker
    ball     (n, 2)    float32, NaN where the ball is not seen
    ball_z   (n,)      float32, height in metres; only filled on ~5% of frames, mostly in the air
    team     (k,)      int8, HOME or AWAY per slot
    blocks   [(half, first_frame, last_frame)] the frames that are played, half by half
    """

    def __init__(self, players, speed, ball, ball_z, team, pitch, blocks=None):
        self.players = players
        self.speed = speed
        self.ball = ball
        self.ball_z = ball_z
        self.team = team
        self.pitch = pitch
        self.blocks = blocks

    @property
    def n(self):
        return len(self.ball)

    def team_mask(self, side):
        return self.team == side


def _atomic_savez(path, **arrays):
    """np.savez through a temporary file, so that parallel runs never read a half-written cache."""
    tmp = "{}.{}.tmp.npz".format(path[:-4], os.getpid())
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def slot_teams(meta_path):
    """HOME/AWAY per track slot. The first team in the metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home = team_ids[0]
    return np.array([HOME if p.get("teamId") == home else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse(raw_path, n_frames, n_slots):
    raw = np.full((n_frames, n_slots + 1, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, players, ball = line.rstrip("\n").split(":")
            row = raw[int(frame_id) - 1]
            for slot, group in enumerate(players.split(";")):
                if group[0] != "N":
                    row[slot] = group.split(",")
            if ball[0] != "N":
                row[n_slots] = [v if v[0] != "N" else "nan" for v in ball.split(",")]
    return raw


def _cache_key(raw_path):
    stat = os.stat(raw_path)
    tag = "{}:{}:{}".format(os.path.abspath(raw_path), stat.st_size, int(stat.st_mtime))
    return hashlib.sha1(tag.encode()).hexdigest()[:12]


def load(game=GAME, data_dir=None, source="tactical", use_cache=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team = slot_teams(meta_path)
    n_frames = source_frames(source, game)
    length, width = GAMES[game]["pitch"]

    cache = os.path.join(CACHE_DIR, "atd_{}_{}.npz".format(source, _cache_key(raw_path)))
    if use_cache and os.path.isfile(cache):
        raw = np.load(cache)["raw"]
    else:
        raw = _parse(raw_path, n_frames, len(team))
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            _atomic_savez(cache, raw=raw)

    scale = np.array([length, width], dtype=np.float32)
    players = raw[:, :-1, :2] * scale
    speed = raw[:, :-1, 2]
    ball = raw[:, -1, :2] * scale
    ball_z = raw[:, -1, 2]
    blocks = half_video_blocks(game) if source == "tactical" else None
    return Tracking(players, speed, ball, ball_z, team, (length, width), blocks)
