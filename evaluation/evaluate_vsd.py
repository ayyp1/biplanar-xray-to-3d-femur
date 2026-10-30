"""External test on VSD. No VSD femur was seen in training, so the 5 fold models are averaged.
The knee offset is learned on all 101 HFValid femurs.

Axis/hip errors are reported against the expert landmarks (19 femurs) and against the landmark
extractor run on the true VSD mask (all femurs, columns *_vs_gtmask). A failure is a predicted hip
more than 20 mm from the true-mask hip.

  python evaluation/evaluate_vsd.py --vsd vsd --data preprocessed --ckpt checkpoints --models attnC swinC
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from femuraxis.common import HFValid, load_models, load_split, predict
from femuraxis.landmarks import fit_head, fit_knee, knee_offset
from femuraxis.metrics import FAIL_HIP_MM, axis_error, overlap, summary

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--vsd", required=True, help="output folder of prepare_vsd.py")
p.add_argument("--data", required=True, help="preprocessed HFValid folder (for the knee offset)")
p.add_argument("--split", default=None)
p.add_argument("--ckpt", default="checkpoints")
p.add_argument("--models", nargs="+", default=["attnC", "swinC"])
p.add_argument("--out", default="results/vsd.csv")
a = p.parse_args()

_, fold_of = load_split(a.split or Path(a.data) / "split.json")
off = knee_offset(HFValid(a.data, sorted(fold_of)), sorted(fold_of))
nets = {m: load_models(m, a.ckpt) for m in a.models}
vsd = Path(a.vsd)
femurs = sorted(d.name for d in vsd.iterdir() if (d / "data.npz").exists())

rows = []
for fid in femurs:
    d = np.load(vsd / fid / "data.npz")
    ap, lat, gt = d["ap"], d["lat"], d["seg"]
    lp = vsd / fid / "landmarks.json"
    lm = json.load(open(lp)) if lp.exists() else None
    hip_gt, knee_gt = fit_head(gt), fit_knee(gt) + off
    if lm:
        rows.append(dict(model="floor", femur=fid, **axis_error(gt, np.array(lm["hip"]), np.array(lm["knee"]), off)))
    for m in a.models:
        pred = predict(nets[m], ap, lat)
        r = dict(model=m, femur=fid, **overlap(pred, gt))
        e = axis_error(pred, hip_gt, knee_gt, off)
        r.update(hip_vs_gtmask_mm=e["hip_mm"], angle_vs_gtmask_deg=e["angle_deg"])
        if lm:
            r.update(axis_error(pred, np.array(lm["hip"]), np.array(lm["knee"]), off))
        r["fail"] = not r["hip_vs_gtmask_mm"] <= FAIL_HIP_MM
        rows.append(r)
    print(fid, flush=True)

df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"\nVSD external test, {len(femurs)} femurs -> {a.out}")
summary(df, ["dice", "hd95", "asd", "nsd2", "hip_vs_gtmask_mm", "angle_vs_gtmask_deg", "hip_mm", "knee_mm", "angle_deg"])
