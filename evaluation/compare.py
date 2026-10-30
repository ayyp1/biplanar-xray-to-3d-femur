"""Paired comparison of two models on the same cases, from a results CSV.

Wilcoxon signed-rank per metric (Holm-corrected), bootstrap 95% CI of the median difference,
McNemar exact test on failures, TOST equivalence on the axis angle, and Spearman Dice vs angle.

  python evaluation/compare.py results/hfvalid_cv.csv --a attnC --b swinC
  python evaluation/compare.py results/vsd.csv --a attnC --b swinC --key femur
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("csv")
p.add_argument("--a", default="attnC")
p.add_argument("--b", default="swinC")
p.add_argument("--key", default="patient")
p.add_argument("--metrics", nargs="+", default=["dice", "hd95", "asd", "nsd2", "hip_mm", "knee_mm", "angle_deg"])
p.add_argument("--margin", type=float, default=0.5, help="TOST equivalence margin for angle_deg (deg)")
p.add_argument("--n-boot", type=int, default=10000)
a = p.parse_args()

df = pd.read_csv(a.csv)
A = df[df.model == a.a].set_index(a.key)
B = df[df.model == a.b].set_index(a.key).reindex(A.index)
rng = np.random.default_rng(0)


def boot_ci(d):
    meds = np.median(rng.choice(d, (a.n_boot, len(d))), axis=1)
    return np.percentile(meds, [2.5, 97.5])


rows = []
for m in a.metrics:
    if m not in A:
        continue
    d = (A[m] - B[m]).dropna().values
    if len(d) < 6:
        continue
    lo, hi = boot_ci(d)
    rows.append(dict(metric=m, n=len(d), median_a=A[m].median(), median_b=B[m].median(),
                     median_diff=np.median(d), ci_lo=lo, ci_hi=hi, p=wilcoxon(d).pvalue))
res = pd.DataFrame(rows)

# Holm step-down
order = np.argsort(res.p.values)
adj = np.minimum(1, np.maximum.accumulate((len(res) - np.arange(len(res))) * res.p.values[order]))
res.loc[res.index[order], "p_holm"] = adj

pd.set_option("display.width", 160)
print(f"{a.a} - {a.b}, paired over {len(A)} cases")
print(res.round(4).to_string(index=False))

if "fail" in A:
    fa, fb = A.fail.astype(bool), B.fail.astype(bool)
    only_a, only_b = int((fa & ~fb).sum()), int((~fa & fb).sum())
    p_mc = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    print(f"\nfailures {int(fa.sum())} vs {int(fb.sum())} (discordant {only_a}/{only_b}), McNemar exact p = {p_mc:.4f}")

if "angle_deg" in A:
    d = (A.angle_deg - B.angle_deg).dropna().values
    p_low = wilcoxon(d + a.margin, alternative="greater").pvalue
    p_high = wilcoxon(d - a.margin, alternative="less").pvalue
    p_tost = max(p_low, p_high)
    print(f"TOST angle_deg, margin +-{a.margin} deg: p = {p_tost:.4g} -> "
          f"{'equivalent' if p_tost < 0.05 else 'equivalence not shown'}")

if {"dice", "angle_deg"} <= set(A.columns):
    for name, g in ((a.a, A), (a.b, B)):
        g = g[["dice", "angle_deg"]].dropna()
        rho, pv = spearmanr(g.dice, g.angle_deg)
        print(f"Spearman Dice vs angle, {name}: rho = {rho:.2f} (p = {pv:.3g})")
