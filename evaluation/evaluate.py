"""HFValid 5-fold cross-validation: every patient is predicted by the fold model that never saw it.

Writes one row per (model, patient). The "floor" rows run the landmark extractor on the true mask,
which is the error of the measurement method alone.

  python evaluation/evaluate.py --data preprocessed --ckpt checkpoints --models attnC swinC
"""
import argparse
from pathlib import Path

import pandas as pd

from femuraxis.common import HFValid, load_models, load_split, predict
from femuraxis.landmarks import knee_offset
from femuraxis.metrics import FAIL_HIP_MM, axis_error, overlap, summary

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True)
p.add_argument("--split", default=None)
p.add_argument("--ckpt", default="checkpoints")
p.add_argument("--models", nargs="+", default=["attnC", "swinC"])
p.add_argument("--out", default="results/hfvalid_cv.csv")
a = p.parse_args()

folds, fold_of = load_split(a.split or Path(a.data) / "split.json")
ids = sorted(fold_of)
data = HFValid(a.data, ids)
# knee offset for each test fold is learned on the other four folds only
offsets = [knee_offset(data, [pid for k, f in enumerate(folds) if k != t for pid in f]) for t in range(len(folds))]
nets = {m: load_models(m, a.ckpt, len(folds)) for m in a.models}

rows = []
for i, pid in enumerate(ids, 1):
    ap, lat = data.drr[pid]
    gt = data.seg[pid]
    hip, knee = data.landmarks(pid)
    off = offsets[fold_of[pid]]
    rows.append(dict(model="floor", patient=pid, **axis_error(gt, hip, knee, off)))
    for m in a.models:
        pred = predict([nets[m][fold_of[pid]]], ap, lat)
        r = dict(model=m, patient=pid, fold=fold_of[pid], **overlap(pred, gt), **axis_error(pred, hip, knee, off))
        r["fail"] = not r["hip_mm"] <= FAIL_HIP_MM
        rows.append(r)
    print(f"[{i}/{len(ids)}] {pid}", flush=True)

df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"\nHFValid 5-fold CV -> {a.out}")
summary(df, ["dice", "hd95", "asd", "nsd1.5", "nsd2", "nsd4", "hip_mm", "knee_mm", "angle_deg"])
