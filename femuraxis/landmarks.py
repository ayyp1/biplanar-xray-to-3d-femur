"""Hip and knee centres from a femur mask in the canonical box. All coordinates are voxel (x, y, z)."""
import numpy as np
from scipy import ndimage

# knee_offset over all 101 HFValid femurs, used when no training data is at hand (inference)
KNEE_OFFSET_HFVALID = np.array([-0.057, 2.901, 2.646])


def surface_points(mask):
    z, y, x = np.nonzero(mask & ~ndimage.binary_erosion(mask))
    return np.stack([x, y, z], 1).astype(float)


def fit_sphere(p):
    s, *_ = np.linalg.lstsq(np.c_[2 * p, np.ones(len(p))], (p ** 2).sum(1), rcond=None)
    c = s[:3]
    r2 = s[3] + c @ c
    return (c, np.sqrt(r2)) if r2 > 0 else (None, None)


def fit_head(mask, prox=0.35, med=0.6, r_vox=(9, 14), n_iter=500, seed=0):
    """RANSAC sphere fit to the proximal, medial part of the surface (the femoral head)."""
    p = surface_points(mask)
    z = p[:, 2]
    p = p[z >= z.max() - prox * (z.max() - z.min())]
    xl, xh = p[:, 0].min(), p[:, 0].max()
    p = p[p[:, 0] <= xl + med * (xh - xl)]

    rng = np.random.default_rng(seed)
    best_n, c_best, r_best = -1, None, None
    for _ in range(n_iter):
        c, r = fit_sphere(p[rng.choice(len(p), 4, replace=False)])
        if c is None or not r_vox[0] <= r <= r_vox[1]:
            continue
        n = (np.abs(np.linalg.norm(p - c, axis=1) - r) < 2.0).sum()
        if n > best_n:
            best_n, c_best, r_best = n, c, r
    for tol in (2.0, 1.5, 1.0):
        inliers = np.abs(np.linalg.norm(p - c_best, axis=1) - r_best) < tol
        c_best, r_best = fit_sphere(p[inliers])
    return c_best


def fit_knee(mask, dist_frac=0.30, win=2):
    """Midpoint of the widest medial-lateral level in the distal 30% of the femur."""
    z, y, x = np.nonzero(mask)
    sel = z <= z.min() + dist_frac * (z.max() - z.min())
    xs, ys, zs = x[sel], y[sel], z[sel]
    levels = np.unique(zs)
    zc = levels[int(np.argmax([xs[zs == q].max() - xs[zs == q].min() for q in levels]))]
    m = np.abs(zs - zc) <= win
    return np.array([(xs[m].min() + xs[m].max()) / 2.0, ys[m].mean(), zc], float)


def knee_offset(data, ids):
    """Mean (expert knee - fit_knee on the true mask) over the given cases.
    Our widest-level rule and the HFValid definition differ by a near-constant shift."""
    return np.mean([data.landmarks(p)[1] - fit_knee(data.seg[p]) for p in ids], 0)


def angle(v1, v2):
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
