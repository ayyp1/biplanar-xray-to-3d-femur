import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDiceMetric, SurfaceDistanceMetric

from .common import VOX
from .landmarks import angle, fit_head, fit_knee

FAIL_HIP_MM = 20.0
NSD_TOL_MM = (1.5, 2.0, 4.0)

_hd95 = HausdorffDistanceMetric(percentile=95)
_asd = SurfaceDistanceMetric(symmetric=True)
_nsd = {t: SurfaceDiceMetric(class_thresholds=[t]) for t in NSD_TOL_MM}


def _t(m):
    return torch.from_numpy(m[None, None].astype(np.float32))


def overlap(pred, gt):
    r = dict(dice=2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1))
    metrics = [("hd95", _hd95), ("asd", _asd)] + [(f"nsd{t:g}", m) for t, m in _nsd.items()]
    for name, m in metrics:
        m.reset()
        m(_t(pred), _t(gt), spacing=(VOX,) * 3)
        r[name] = float(m.aggregate())
    return r


def axis_error(mask, hip_ref, knee_ref, knee_off):
    """Hip/knee centre error (mm) and mechanical axis angle error (deg) of a mask against reference points."""
    try:
        hip, knee = fit_head(mask), fit_knee(mask) + knee_off
    except Exception:
        return dict(hip_mm=np.nan, knee_mm=np.nan, angle_deg=np.nan)
    return dict(hip_mm=np.linalg.norm(hip - hip_ref) * VOX,
                knee_mm=np.linalg.norm(knee - knee_ref) * VOX,
                angle_deg=angle(hip - knee, hip_ref - knee_ref))


def med_iqr(s):
    s = s.dropna()
    return f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]" if len(s) else "-"


def summary(df, metrics, by="model"):
    names = list(dict.fromkeys(df[by]))
    print(f"{'median [IQR]':16}" + "".join(f"{n:>24}" for n in names))
    for c in metrics:
        if c in df:
            print(f"{c:16}" + "".join(f"{med_iqr(df[df[by] == n][c]):>24}" for n in names))
    for n in names:
        g = df[df[by] == n]
        if "fail" in g and g.fail.notna().any():
            a = g.angle_deg.dropna()
            print(f"{n}: failures {int(g.fail.sum())}/{len(g)} | <=1/2/3 deg "
                  f"{(a <= 1).mean():.0%}/{(a <= 2).mean():.0%}/{(a <= 3).mean():.0%} | worst {a.max():.2f} deg")
