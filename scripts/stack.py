"""Second stage: learn the temporal shape of the first-stage probabilities.

    python scripts/stack.py RUN_DIR [--write]

RUN_DIR holds the held-out predictions train.py writes for every DFL match (heldout_*.npz): the
first-stage probabilities on a match from models that never saw it. A second set of classifiers
reads, for each frame, those probabilities at offsets up to two seconds either side and in
windows around it, and is cross-validated the same way, leave one match out. Its output feeds the
same decoder. With --write the second-stage models go into models/ next to the first.
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train  # noqa: E402
from possession import learned  # noqa: E402

GRID = {
    "switch_ha": [4.0, 6.0, 9.0, 13.0],
    "switch_dead": [4.0, 8.0, 12.0],
    "switch_restarter": [1.0],
}


def load_run(folder):
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "heldout_*.npz"))):
        z = np.load(path)
        name = os.path.basename(path)[8:-4]
        out[name] = {"p": (z["p_dead"], z["p_home"], z["p_restart"]), "y": z["labels"],
                     "next": learned.next_restarter(z["labels"]),
                     "blocks": [tuple(int(v) for v in b) for b in z["blocks"]]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    data = load_run(args.run)
    matches = sorted(data)
    feats = {m: learned.stack_features(*data[m]["p"]) for m in matches}
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]

    table = np.zeros((len(matches), len(combos), 2, 4))      # [match, combo, stage, metric]
    for i, held in enumerate(matches):
        parts = [{"x": feats[m], "y": data[m]["y"], "next": data[m]["next"]}
                 for m in matches if m != held]
        x, y = train.stack(parts)
        models = learned.fit(x, y)
        stage2 = learned.predict(models, feats[held])
        for stage, probs in enumerate((data[held]["p"], stage2)):
            scores = learned.emissions(*probs)
            for j, params in enumerate(combos):
                pred = learned.decode(scores, data[held]["blocks"], params)
                table[i, j, stage] = train.score(data[held]["y"], pred, data[held]["blocks"])
        print("{}  stage 1 best {:.3f}  stage 2 best {:.3f}".format(
            held, table[i, :, 0, 0].max(), table[i, :, 1, 0].max()), flush=True)

    for stage in (0, 1):
        mean = table[:, :, stage, 0].mean(axis=0)
        j = int(np.argmax(mean))
        print("stage {}: mean spellF1 {:.3f} macroF1 {:.3f} at {}".format(
            stage + 1, mean[j], table[:, j, stage, 1].mean(), combos[j]))

    if args.write:
        parts = [{"x": feats[m], "y": data[m]["y"], "next": data[m]["next"]} for m in matches]
        x, y = train.stack(parts)
        learned.save({"stack_" + k: v for k, v in learned.fit(x, y).items()})
        print("wrote second-stage models to", learned.MODEL_DIR)


if __name__ == "__main__":
    main()
