"""Train the per-frame models on the DFL matches and pick the decoder costs.

    python scripts/get_dfl.py          # once, ~2.6 GB
    python scripts/train.py            # cross-validation, then the final models into models/
    python scripts/train.py --cv-only  # just the leave-one-match-out table

Each DFL match is degraded (possession/degrade.py) so that it looks like the competition's
automatic tracking, COPIES times with different draws, every other copy with the sides swapped. Decoder costs are chosen by spellF1 on
held-out matches: every match is scored by models that never saw it. Nothing here touches the
competition data.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import degrade, dfl, framefeatures, learned  # noqa: E402
from possession.features import Features  # noqa: E402

COPIES = 2
SUBSAMPLE = 2          # every 2nd frame for training; neighbours carry almost the same information
GRID = {
    "switch_ha": [2.0, 4.0, 6.0, 9.0, 13.0],
    "switch_dead": [2.0, 4.0, 8.0],
    "switch_restarter": [1.0, 3.0],
}


def build(match_id, seed, swap=False):
    """One degraded copy of a match. swap=True relabels home as away and the other way round, so
    that over the copies neither side is special and the models cannot learn a home prior."""
    t = dfl.load(match_id)
    labels = t.labels
    if swap:
        t = dfl.swap_sides(t)
        labels = t.labels
    rng = np.random.default_rng(seed)
    noisy, params = degrade.degrade(t, labels, rng)
    x, names = framefeatures.compute(noisy, Features(noisy))
    return {"x": x, "y": labels, "next": learned.next_restarter(labels),
            "blocks": t.blocks, "names": names, "params": params}


def stack(parts):
    xs, ys, ns = [], [], []
    for p in parts:
        rows = np.flatnonzero(p["y"] != 3)[::SUBSAMPLE]
        xs.append(p["x"][rows])
        ys.append(p["y"][rows])
        ns.append(p["next"][rows])
    return np.concatenate(xs), (np.concatenate(ys), np.concatenate(ns))


def score(labels, pred, blocks):
    gt = labels.astype(np.int8)
    pred = np.where(pred < 0, 3, pred).astype(np.int8)
    graded = np.zeros(len(gt), dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    gt = np.where(graded, gt, 3).astype(np.int8)
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--skip-cv", action="store_true",
                        help="train the final models with the decoder costs in learned.DECODER")
    parser.add_argument("--matches", nargs="*")
    parser.add_argument("--copies", type=int, default=COPIES,
                        help="degraded copies of each match (default %(default)s)")
    parser.add_argument("--models", default=learned.MODEL_DIR, help="where to write the models")
    parser.add_argument("--gbm", default="", help="overrides for learned.GBM, e.g. max_iter=600")
    args = parser.parse_args()
    for item in filter(None, args.gbm.split(",")):
        key, value = item.split("=")
        learned.GBM[key] = type(learned.GBM[key])(value)

    matches = args.matches or dfl.match_ids()
    t0 = time.time()
    data = {m: [build(m, seed=1000 * i + c, swap=bool(c % 2)) for c in range(args.copies)]
            for i, m in enumerate(matches)}
    print("features built for {} matches in {:.0f} s".format(len(matches), time.time() - t0))

    if args.skip_cv:
        finish(data, matches, dict(learned.DECODER), None, args.models)
        return

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    table = np.zeros((len(matches), len(combos), 4))
    for i, held in enumerate(matches):
        x, y = stack([p for m in matches if m != held for p in data[m]])
        models = learned.fit(x, y)
        test = data[held][0]
        probs = learned.predict(models, test["x"])
        os.makedirs(args.models, exist_ok=True)
        np.savez_compressed(os.path.join(args.models, "heldout_{}.npz".format(held)),
                            p_dead=probs[0], p_home=probs[1], p_restart=probs[2],
                            labels=test["y"], blocks=np.array(test["blocks"]))
        scores = learned.emissions(*probs)
        for j, params in enumerate(combos):
            pred = learned.decode(scores, test["blocks"], params)
            table[i, j] = score(test["y"], pred, test["blocks"])
        best = int(np.argmax(table[i, :, 0]))
        print("{}  best spellF1 {:.3f} macroF1 {:.3f} poss {:.0f}/{:.0f} at {}".format(
            held, *table[i, best], combos[best]), flush=True)

    mean = table[:, :, 0].mean(axis=0)
    order = np.argsort(-mean)
    print("\nmean over held-out matches")
    for j in order[:8]:
        print("  spellF1 {:.3f}  macroF1 {:.3f}  poss ratio {:.2f}  {}".format(
            mean[j], table[:, j, 1].mean(), (table[:, j, 2] / table[:, j, 3]).mean(), combos[j]))
    best = int(order[0])
    print("\nper match at {}".format(combos[best]))
    for i, m in enumerate(matches):
        print("  {}  spellF1 {:.3f}  macroF1 {:.3f}  possessions {:.0f} / {:.0f}".format(m, *table[i, best]))
    chosen = dict(learned.DECODER, **combos[best])
    if not args.cv_only:
        finish(data, matches, chosen, float(mean[best]), args.models)


def finish(data, matches, decoder, cv_score, folder):
    x, y = stack([p for m in matches for p in data[m]])
    models = learned.fit(x, y)
    learned.save(models, folder)
    with open(os.path.join(folder, "decoder.json"), "w") as handle:
        json.dump({"decoder": decoder, "features": data[matches[0]][0]["names"],
                   "cv_spell_f1": cv_score, "matches": matches}, handle, indent=1)
    print("saved models and decoder costs {} to {}".format(decoder, folder))


if __name__ == "__main__":
    main()
