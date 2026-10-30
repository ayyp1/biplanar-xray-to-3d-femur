"""Prepare the VSD lower-body database (Fischer 2023) as an external test set, exactly like HFValid.

CT + femur/patella meshes come from Zenodo (10.5281/zenodo.8302448, CC BY-NC-SA 4.0). Only the needed
files are read out of each zip with HTTP range requests, so the full ~40 GB is never downloaded.
Expert landmarks (5 raters) come from github.com/MCM-Fischer/VSDFullBodyBoneModels.

Per femur in <out>/<subject>_<side>/: data.npz (ap, lat with row = z, solid seg) and, when experts
exist, landmarks.json (hip, knee in canonical voxel coordinates). Plus prepare_log.csv and contact_sheet.png.

The VSD headers are not consistent between subjects, so the orientation is set from the anatomy:
femoral head up, patella in front, left femur at +x of the right one.

  python preprocessing/prepare_vsd.py --out vsd
"""
import argparse
import io
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import SimpleITK as sitk
import vtk
from scipy import ndimage
from vtk.util import numpy_support

from femuraxis.common import DRR_SCALE, VOX

BOX = (96, 128, 256)  # (x, y, z), as in preprocess_hfvalid.py
CROP_MM = dict(medial=30, lateral=34, below=16, above=40)  # HFValid scan coverage around the femur (medians)
SIDE_MIRROR = {"L": np.array([1, 1, 1]), "R": np.array([-1, 1, 1])}
ZENODO = "https://zenodo.org/api/records/8302448"
RATER_URL = "https://raw.githubusercontent.com/MCM-Fischer/VSDFullBodyBoneModels/master/ManualLandmarks/Femur/Rater{}.mat"


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so zipfile can read single members."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def extract(z, name, tmp):
    path = Path(tmp) / Path(name).name
    with z.open(name) as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 << 20)
    return path


def read_ply(path):
    r = vtk.vtkPLYReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()


def points(mesh):
    return numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)


def mesh_to_mask(ct, mesh, pts):
    """Same as stl_to_mask.py, with the mesh points given explicitly (after the frame flip)."""
    o, sp, D = np.array(ct.GetOrigin()), np.array(ct.GetSpacing()), np.array(ct.GetDirection()).reshape(3, 3)
    idx = (pts - o) @ np.linalg.inv(D @ np.diag(sp)).T
    m = vtk.vtkPolyData()
    m.DeepCopy(mesh)
    m.GetPoints().SetData(numpy_support.numpy_to_vtk(idx, deep=True))
    size = ct.GetSize()
    st = vtk.vtkPolyDataToImageStencil()
    st.SetInputData(m)
    st.SetOutputOrigin(0, 0, 0)
    st.SetOutputSpacing(1, 1, 1)
    st.SetOutputWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    st.Update()
    im = vtk.vtkImageStencilToImage()
    im.SetInputConnection(st.GetOutputPort())
    im.SetInsideValue(1)
    im.SetOutsideValue(0)
    im.SetOutputScalarTypeToUnsignedChar()
    im.Update()
    arr = numpy_support.vtk_to_numpy(im.GetOutput().GetPointData().GetScalars()).reshape(size[::-1])
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct)
    return out


def fill_solid(seg):
    # the raw PLY meshes include the inner marrow-canal surface, which leaves the shaft hollow
    s = ndimage.binary_fill_holes(seg)
    return np.stack([ndimage.binary_fill_holes(sl) for sl in s])


def mesh_flip(ct, pts):
    """Mesh frame -> CT frame: identity or a 180 deg rotation (some subjects are stored rotated).
    The candidate that puts the mesh inside the CT with the highest HU on its surface wins."""
    o, sp, n = np.array(ct.GetOrigin()), np.array(ct.GetSpacing()), np.array(ct.GetSize())
    lo, hi = o, o + sp * (n - 1)
    arr = sitk.GetArrayViewFromImage(ct)
    best = None
    for s in map(np.array, ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1])):
        q = pts * s
        if ((q >= lo) & (q <= hi)).all(1).mean() < 0.95:
            continue
        i = np.clip(np.round((q[::50] - lo) / sp).astype(int), 0, n - 1)
        hu = float(np.median(arr[i[:, 2], i[:, 1], i[:, 0]]))
        if best is None or hu > best[1]:
            best = (s, hu)
    return best


def anatomical_signs(fem_l, fem_r, patella):
    sx = 1 if fem_l[:, 0].mean() > fem_r[:, 0].mean() else -1
    sy = 1 if patella[:, 1].mean() < fem_l[:, 1].mean() else -1
    sz = 1 if patella[:, 2].mean() < fem_l[:, 2].mean() else -1
    return np.array([sx, sy, sz])


def sphere_center(p):
    s, *_ = np.linalg.lstsq(np.c_[2 * p, np.ones(len(p))], (p ** 2).sum(1), rcond=None)
    return s[:3]


def expert_landmarks(raters, subj, side):
    """Hip = sphere through the 6 femoral-head points, knee = epicondyle midpoint; mean over raters and trials."""
    hips, knees = [], []
    for res in raters:
        row = [r for r in res if str(r[4][0]) == subj and str(r[5][0]) == side]
        if not row:
            continue
        for t in range(4):
            c = row[0][t]
            if not hasattr(c, "shape") or c.shape != (22, 3):
                continue
            pt = {str(c[i, 0][0]).split("(")[-1].rstrip(")"): np.asarray(c[i, 1], float).ravel() for i in range(22)}
            hips.append(sphere_center(np.array([pt[k] for k in ("SFH", "AFH", "LFH", "PFH", "MFH", "IFH")])))
            knees.append((pt["MEC"] + pt["LEC"]) / 2)
    if not hips:
        return None
    hips, knees = np.array(hips), np.array(knees)
    return dict(hip=hips.mean(0), knee=knees.mean(0), n=len(hips),
                hip_spread_mm=float(np.median(np.linalg.norm(hips - hips.mean(0), axis=1))),
                knee_spread_mm=float(np.median(np.linalg.norm(knees - knees.mean(0), axis=1))))


def process_subject(subj, url, raters, out):
    with tempfile.TemporaryDirectory() as tmp:
        z = zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=16 << 20))
        names = z.namelist()
        get = lambda suffix: next(n for n in names if n.endswith(suffix))
        meshes = {s: read_ply(extract(z, get(f"Models/Femur_{s}.ply"), tmp)) for s in "LR"}
        patella = points(read_ply(extract(z, get("Models/Patella_L.ply"), tmp)))
        both = np.vstack([points(meshes["L"]), points(meshes["R"])])
        for n in (n for n in names if n.endswith(".nrrd") and not n.endswith(".seg.nrrd")):
            ct = sitk.ReadImage(str(extract(z, n, tmp)))
            flip = mesh_flip(ct, both)
            if flip is not None:
                break
        else:
            raise RuntimeError("no CT volume contains the femurs")
        s_ct, surface_hu = flip
        s_anat = anatomical_signs(points(meshes["L"]), points(meshes["R"]), patella)

        rows = []
        for side in "LR":
            pts = points(meshes[side]) * s_ct
            mask = mesh_to_mask(ct, meshes[side], pts)
            s_can = s_ct * s_anat * SIDE_MIRROR[side]
            k, j, i = np.nonzero(sitk.GetArrayViewFromImage(mask))
            corners = np.array([s_can * np.array(mask.TransformIndexToPhysicalPoint((int(x), int(y), int(zz))))
                                for x in (i.min(), i.max()) for y in (j.min(), j.max()) for zz in (k.min(), k.max())])
            origin = (corners.min(0) + corners.max(0)) / 2 - VOX * (np.array(BOX) - 1) / 2
            ref = sitk.Image(list(BOX), sitk.sitkUInt8)
            ref.SetSpacing([VOX] * 3)
            ref.SetOrigin(origin.tolist())
            tf = sitk.AffineTransform(3)
            tf.SetMatrix(np.diag(s_can.astype(float)).ravel().tolist())
            hu = sitk.GetArrayFromImage(sitk.Resample(ct, ref, tf, sitk.sitkLinear, -1024.0, sitk.sitkFloat32))
            seg = sitk.GetArrayFromImage(sitk.Resample(mask, ref, tf, sitk.sitkLinear, 0.0, sitk.sitkFloat32)) >= 0.5

            # crop the scan like HFValid; this also removes the other leg
            zz, _, xx = np.nonzero(seg)
            v = lambda mm: int(round(mm / VOX))
            keep = np.zeros_like(seg)
            keep[max(zz.min() - v(CROP_MM["below"]), 0):zz.max() + v(CROP_MM["above"]) + 1, :,
                 max(xx.min() - v(CROP_MM["medial"]), 0):xx.max() + v(CROP_MM["lateral"]) + 1] = True
            hu[~keep] = -1024.0
            mu = np.clip(0.02 * (1 + hu / 1000.0), 0, None)
            ap, lat = mu.sum(axis=1) * VOX, mu.sum(axis=2) * VOX

            fid = f"{subj}_{side}"
            d = out / fid
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "data.npz", ap=np.clip(ap / DRR_SCALE, 0, 1).astype(np.float32),
                                lat=np.clip(lat / DRR_SCALE, 0, 1).astype(np.float32), seg=fill_solid(seg))
            ex = expert_landmarks(raters, subj, side)
            if ex:
                lm = {k: ((SIDE_MIRROR[side] * ex[k] - origin) / VOX).tolist() for k in ("hip", "knee")}
                json.dump(lm | {k: ex[k] for k in ("n", "hip_spread_mm", "knee_spread_mm")}, open(d / "landmarks.json", "w"))
            touches = any(f.any() for f in (seg[0], seg[-1], seg[:, 0], seg[:, -1], seg[:, :, 0], seg[:, :, -1]))
            rows.append(dict(femur=fid, subject=subj, side=side, ct=Path(n).name, mesh_flip=str(s_ct), anat=str(s_anat),
                             surface_hu=surface_hu, median_hu_in_mask=float(np.median(hu[seg])),
                             femur_len_mm=float((zz.max() - zz.min() + 1) * VOX), touches_border=touches,
                             drr_max=float(max(ap.max(), lat.max())), expert=ex is not None))
        return rows


def contact_sheet(out, fids, path, cols=10):
    rows = int(np.ceil(len(fids) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.6), squeeze=False)
    for a in axs.flat:
        a.axis("off")
    for a, fid in zip(axs.flat, fids):
        d = np.load(out / fid / "data.npz")
        gap = np.zeros((d["ap"].shape[0], 4))
        a.imshow(np.concatenate([d["ap"], gap, d["lat"]], 1), cmap="gray", origin="lower", vmin=0, vmax=1)
        a.contour(np.concatenate([d["seg"].any(1), gap, d["seg"].any(2)], 1), levels=[0.5], colors="r", linewidths=0.6)
        lp = out / fid / "landmarks.json"
        if lp.exists():
            lm = json.load(open(lp))
            for k, c in (("hip", "lime"), ("knee", "magenta")):
                a.plot([lm[k][0], lm[k][1] + d["ap"].shape[1] + 4], [lm[k][2]] * 2, "+", c=c, ms=6, mew=1.5, ls="none")
        a.set_title(fid, fontsize=7)
    fig.suptitle("VSD prepared like HFValid (AP | LAT): red = femur, + = expert hip (green) / knee (magenta)")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="vsd")
    p.add_argument("--subjects", nargs="*", help="e.g. z001 z002 (default: all)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = json.load(urllib.request.urlopen(ZENODO))
    zips = {f["key"][:-4]: f["links"]["self"] for f in rec["files"]
            if f["key"].endswith(".zip") and not f["key"].startswith("Phantom")}
    raters = []
    for r in range(1, 6):
        path = out / f"Rater{r}.mat"
        if not path.exists():
            urllib.request.urlretrieve(RATER_URL.format(r), path)
        raters.append(scipy.io.loadmat(path)["Results"])

    log_p = out / "prepare_log.csv"
    log = pd.read_csv(log_p).to_dict("records") if log_p.exists() else []
    done = {r["subject"] for r in log}
    for subj in a.subjects or sorted(zips):
        if subj in done:
            continue
        t0 = time.time()
        try:
            log += process_subject(subj, zips[subj], raters, out)
            print(f"{subj}: ok ({time.time() - t0:.0f}s), median HU in mask "
                  f"{log[-2]['median_hu_in_mask']:.0f}/{log[-1]['median_hu_in_mask']:.0f}", flush=True)
        except Exception as e:
            print(f"{subj}: FAILED {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(log).to_csv(log_p, index=False)

    log = pd.DataFrame(log)
    contact_sheet(out, sorted(log.femur), out / "contact_sheet.png")
    print(f"femurs: {len(log)} | with experts: {int(log.expert.sum())} | touching border: {int(log.touches_border.sum())}"
          f" | DRR above scale: {int((log.drr_max > DRR_SCALE).sum())}")
    print(f"check {out / 'contact_sheet.png'}: femoral head at the top, head on the left of the AP image")


if __name__ == "__main__":
    main()
