"""Test-time robustness on HFValid: the DRRs are degraded before prediction, the truth is unchanged.
Each patient is still predicted by its own held-out fold model.

  python evaluation/robustness.py --data preprocessed --ckpt checkpoints --models attnC swinC
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from femuraxis.common import HFValid, load_models, load_split, predict, project
from femuraxis.landmarks import knee_offset
from femuraxis.metrics import FAIL_HIP_MM, axis_error, summary

CONDITIONS = [("none", 0), ("noise", 0.05), ("blur", 3), ("gamma", 0.7), ("gamma", 1.4), ("lat_angle", 5), ("lat_angle", 10)]


def degrade(data, pid, kind, level, seed):
    ap, lat = (d.copy() for d in data.drr[pid])
    rng = np.random.default_rng([seed, int(round(level * 1000))])
    if kind == "noise":
        ap, lat = (np.clip(d + rng.normal(0, level, d.shape), 0, 1) for d in (ap, lat))
    elif kind == "blur":
        ap, lat = (ndimage.gaussian_filter(d, level) for d in (ap, lat))
    elif kind == "gamma":
        ap, lat = (np.clip(d, 0, 1) ** level for d in (ap, lat))
    elif kind == "lat_angle":
        # side view taken from a CT rotated about the long axis, i.e. a non-orthogonal second view
        hu = ndimage.rotate(data.ct(pid).astype(np.float32), level, axes=(1, 2), reshape=False, order=1, cval=-1024)
        lat = project(hu)[1]
    return ap.astype(np.float32), lat.astype(np.float32)


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True)
p.add_argument("--split", default=None)
p.add_argument("--ckpt", default="checkpoints")
p.add_argument("--models", nargs="+", default=["attnC", "swinC"])
p.add_argument("--out", default="results/robustness.csv")
a = p.parse_args()

folds, fold_of = load_split(a.split or Path(a.data) / "split.json")
ids = sorted(fold_of)
data = HFValid(a.data, ids)
offsets = [knee_offset(data, [pid for k, f in enumerate(folds) if k != t for pid in f]) for t in range(len(folds))]
nets = {m: load_models(m, a.ckpt, len(folds)) for m in a.models}

rows = []
for kind, level in CONDITIONS:
    for i, pid in enumerate(ids):
        ap, lat = degrade(data, pid, kind, level, i)
        gt = data.seg[pid]
        hip, knee = data.landmarks(pid)
        for m in a.models:
            pred = predict([nets[m][fold_of[pid]]], ap, lat)
            r = dict(model=m, condition=f"{kind} {level:g}", patient=pid,
                     dice=2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1),
                     **axis_error(pred, hip, knee, offsets[fold_of[pid]]))
            r["fail"] = not r["hip_mm"] <= FAIL_HIP_MM
            rows.append(r)
    print(f"\n== {kind} {level:g}")
    g = pd.DataFrame(rows)
    summary(g[g.condition == f"{kind} {level:g}"], ["dice", "hip_mm", "angle_deg"])

df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"\n-> {a.out}")
