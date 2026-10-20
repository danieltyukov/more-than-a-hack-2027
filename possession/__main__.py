"""python -m possession [--out PATH] [--no-broadcast]

Writes the submission from the data in ./data (or $MTAH_DATA_DIR). Needs no training: the models
are in models/. The output is checked with the same validation the official scorer runs.
"""

import argparse
import os
import sys
import time

import numpy as np

from .atd import ROOT
from .pipeline import run, write_submission

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession.csv")


def main():
    parser = argparse.ArgumentParser(prog="python -m possession")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-broadcast", action="store_true",
                        help="ignore the broadcast recording (scores a little worse)")
    args = parser.parse_args()

    start = time.time()
    labels = run(use_broadcast=not args.no_broadcast)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_submission(args.out, labels)

    problem = pm.validate_submission(pm.read_label_csv(args.out), len(labels))
    if problem:
        sys.exit("written but INVALID: {}".format(problem))
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}  ({:.0f} s)".format(
        args.out, time.time() - start, **counts))


if __name__ == "__main__":
    main()
