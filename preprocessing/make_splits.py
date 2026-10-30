"""5-fold patient split of the preprocessed HFValid cases (reproduces split.json with seed 42)."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("data", help="preprocessed folder")
p.add_argument("--out", default=None, help="default: <data>/split.json")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

ids = sorted(d.name for d in Path(a.data).iterdir() if d.is_dir() and d.name != "qa")
folds = [list(f) for f in np.array_split(np.random.default_rng(a.seed).permutation(ids), a.folds)]
out = Path(a.out or Path(a.data) / "split.json")
json.dump(folds, open(out, "w"), indent=2)
print(f"{len(ids)} patients -> folds of {[len(f) for f in folds]} -> {out}")
