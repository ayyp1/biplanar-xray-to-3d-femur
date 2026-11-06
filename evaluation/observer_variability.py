"""VSD observer variability: each expert observation (5 raters x 4 repetitions, 19 femurs) against the consensus.
Hip = sphere through the six head points, knee = (MEC + LEC) / 2, as in the VSD protocol.
  python observer_variability.py <folder with Rater1.mat .. Rater5.mat>
  (github.com/MCM-Fischer/VSDFullBodyBoneModels, ManualLandmarks/Femur)
"""
import re
import sys

import numpy as np
import scipy.io as sio


def sphere(p):
    s, *_ = np.linalg.lstsq(np.c_[2 * p, np.ones(len(p))], (p ** 2).sum(1), rcond=None)
    return s[:3]


obs = {}
for r in range(1, 6):
    for row in sio.loadmat(f"{sys.argv[1]}/Rater{r}.mat", simplify_cells=True)["Results"]:
        for t in range(4):
            P = {re.search(r"\((\w+)\)", c[0]).group(1): np.asarray(c[1], float) for c in row[t]}
            hip = sphere(np.array([P[k] for k in ("SFH", "AFH", "LFH", "PFH", "MFH", "IFH")]))
            obs.setdefault((row[4], row[5]), []).append((r, hip, (P["MEC"] + P["LEC"]) / 2))

ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
for level in ("observation", "rater mean"):
    out = []
    for L in obs.values():
        rater = np.array([o[0] for o in L]); H = np.array([o[1] for o in L]); K = np.array([o[2] for o in L])
        hc, kc = H.mean(0), K.mean(0); ax = hc - kc; u = ax / np.linalg.norm(ax)
        if level == "rater mean":
            H = np.array([H[rater == q].mean(0) for q in range(1, 6)]); K = np.array([K[rater == q].mean(0) for q in range(1, 6)])
        for h, k in zip(H, K):
            eh, ek = h - hc, k - kc
            out.append((np.linalg.norm(eh), np.linalg.norm(ek), abs(ek @ u), np.linalg.norm(ek - (ek @ u) * u), ang(h - k, ax)))
    X = np.array(out); m = np.median(X, 0)
    print(f"{level} vs consensus (n={len(X)}): hip {m[0]:.2f} mm, knee {m[1]:.2f} mm (along {m[2]:.2f}, across {m[3]:.2f}), "
          f"axis {m[4]:.2f} deg (p95 {np.percentile(X[:, 4], 95):.2f})")

# observer vs observer: pairs of single observations by two different raters (what a method scored against one
# reference annotation should be compared with)
out = []
for L in obs.values():
    H = np.array([o[1] for o in L]); K = np.array([o[2] for o in L]); ax = H.mean(0) - K.mean(0); u = ax / np.linalg.norm(ax)
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if L[i][0] == L[j][0]: continue
            ek = K[i] - K[j]
            out.append((np.linalg.norm(H[i] - H[j]), np.linalg.norm(ek), abs(ek @ u), np.linalg.norm(ek - (ek @ u) * u),
                        ang(H[i] - K[i], H[j] - K[j])))
X = np.array(out); m = np.median(X, 0)
print(f"observer vs observer, different raters (n={len(X)} pairs): hip {m[0]:.2f} mm, knee {m[1]:.2f} mm (along {m[2]:.2f}, "
      f"across {m[3]:.2f}), axis {m[4]:.2f} deg (p95 {np.percentile(X[:, 4], 95):.2f})")
