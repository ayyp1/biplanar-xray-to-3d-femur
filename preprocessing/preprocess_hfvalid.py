"""
Preprocess HFValid: CT + expert STL -> canonical box (all femurs as LEFT), DRRs, landmarks.

Per patient, in <out>/<PatID>/:  ct.nii.gz, seg.nii.gz, ap.png, lat.png, landmarks.json
Plus <out>/dataset.csv and <out>/qa/ (per-patient overlays + contact_sheet.png).

Conventions (canonical space):
  - LPS physical axes, direction = identity. Right femurs are mirrored in x: q = diag(-1,1,1) p.
  - Box: --box (x, y, z) voxels of --vox mm, centred on the femur mask's bounding box.
    Default 96 x 128 x 256 at 2 mm = 192 x 256 x 512 mm (long axis = z, hip to knee;
    y is wider because flexed hips make the femur lie diagonally front-to-back, e.g. Pat007).
  - numpy arrays are (z, y, x) = (S, P, L).
  - DRR = parallel projection: mu = max(0, 0.02 (1 + HU/1000)) per mm, L = sum(mu) * vox.
    AP = sum over y -> image (z, x);  LAT = sum over x -> image (z, y).
    Rows flipped: row 0 = highest z (hip at top).
    Landmark voxel (x, y, z) -> AP pixel (col=x, row=bz-1-z), LAT pixel (col=y, row=bz-1-z), bz = box z size.
  - PNGs are 16-bit: value = clip(L / scale, 0, 1) * 65535, one fixed scale for all patients.

Usage:
  python preprocessing/preprocess_hfvalid.py /path/to/HFValid_Collection_v3 [OUT] [--patients Pat001 Pat002]
  python preprocessing/preprocess_hfvalid.py --selftest      # synthetic left + right femur
"""
import argparse
import csv
import json
import re
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

from stl_to_mask import read_stl, stl_to_mask

LANDMARKS = ["femurHead", "kneeCenter", "headDir", "greatTroch"]
COLORS = {"femurHead": "lime", "kneeCenter": "magenta"}


def read_inp(path):
    vals = {}
    for line in open(path):
        m = re.match(r"\s*(\w+?)([XYZ])\s*=\s*([-+0-9.eE]+)", line)
        if m:
            vals.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    return {k: np.array([v["X"], v["Y"], v["Z"]]) for k, v in vals.items()}


def read_sides(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        return {r["PatientID"]: r["Side"].strip() for r in csv.DictReader(f, delimiter=";")}


def to_png16(img, scale):
    return (np.clip(img / scale, 0, 1) * 65535).astype(np.uint16)


def process(d, pid, side, out, box, vox, scale):
    ct = sitk.ReadImage(str(d / f"{pid}.nrrd"))
    mask = stl_to_mask(ct, read_stl(str(d / f"{pid}.stl")))
    lm = read_inp(d / f"{pid}_palp.inp")
    mir = np.array([-1.0 if side == "R" else 1.0, 1.0, 1.0])

    # Canonical box centred on the (mirrored) mask's bounding box
    k, j, i = np.nonzero(sitk.GetArrayViewFromImage(mask))
    if len(k) == 0:
        raise ValueError("empty mask (STL outside CT?)")
    corners = np.array([mir * np.array(mask.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
                        for x in (i.min(), i.max()) for y in (j.min(), j.max()) for z in (k.min(), k.max())])
    origin = (corners.min(0) + corners.max(0)) / 2 - vox * (np.array(box) - 1) / 2

    ref = sitk.Image(list(box), sitk.sitkUInt8)  # size (x, y, z); direction = identity
    ref.SetSpacing([vox] * 3)
    ref.SetOrigin(origin.tolist())
    tf = sitk.AffineTransform(3)
    tf.SetMatrix(np.diag(mir).ravel().tolist())  # maps output point -> input point; a mirror is its own inverse
    ct_c = sitk.Resample(ct, ref, tf, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    # linear + threshold, not nearest neighbour: NN breaks exact half-voxel ties in index direction,
    # which shifts mirrored (right) femurs by one voxel relative to left ones
    seg_c = sitk.Resample(mask, ref, tf, sitk.sitkLinear, 0.0, sitk.sitkFloat32) >= 0.5

    hu = sitk.GetArrayFromImage(ct_c)
    seg = sitk.GetArrayFromImage(seg_c).astype(bool)
    touches = any(f.any() for f in (seg[0], seg[-1], seg[:, 0], seg[:, -1], seg[:, :, 0], seg[:, :, -1]))

    mu = np.clip(0.02 * (1 + hu / 1000.0), 0, None)
    ap = mu.sum(axis=1)[::-1] * vox
    lat = mu.sum(axis=2)[::-1] * vox

    lmc = {n: {"mm": (mir * p).tolist(), "voxel": ((mir * p - origin) / vox).tolist()} for n, p in lm.items()}

    po = out / pid
    po.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(ct_c, sitk.sitkInt16), str(po / "ct.nii.gz"), useCompression=True)
    sitk.WriteImage(seg_c, str(po / "seg.nii.gz"), useCompression=True)
    sitk.WriteImage(sitk.GetImageFromArray(to_png16(ap, scale)), str(po / "ap.png"))
    sitk.WriteImage(sitk.GetImageFromArray(to_png16(lat, scale)), str(po / "lat.png"))
    with open(po / "landmarks.json", "w") as f:
        json.dump({"patient": pid, "side": side, "box": list(box), "vox_mm": vox,
                   "origin_mm": origin.tolist(), "landmarks": lmc}, f, indent=2)

    print(f"{pid} ({side}): {seg.sum()} mask voxels, L max AP {ap.max():.2f} LAT {lat.max():.2f}"
          + ("   WARNING: femur touches box border" if touches else ""))

    thumb = (pid, side, ap, lat, seg.any(axis=1)[::-1], seg.any(axis=2)[::-1], lmc)
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    draw(ax, thumb, box, scale)
    fig.savefig(out / "qa" / f"{pid}.png", dpi=120)
    plt.close(fig)

    row = {"patient": pid, "side": side, "ct": f"{pid}/ct.nii.gz", "seg": f"{pid}/seg.nii.gz",
           "ap": f"{pid}/ap.png", "lat": f"{pid}/lat.png", "landmarks": f"{pid}/landmarks.json",
           "touches_border": touches, "ap_max": round(float(ap.max()), 3), "lat_max": round(float(lat.max()), 3)}
    for n, v in lmc.items():
        for c, val in zip("xyz", v["mm"]):
            row[f"{n}_{c}"] = round(val, 3)
    return row, thumb


def draw(ax, thumb, box, scale):
    """AP + LAT with projected mask outline (red) and projected hip/knee centres."""
    pid, side, ap, lat, apm, latm, lmc = thumb
    for col, (a, img, m) in enumerate(((ax[0], ap, apm), (ax[1], lat, latm))):
        a.imshow(img, cmap="gray", vmin=0, vmax=scale)
        a.contour(m, levels=[0.5], colors="r", linewidths=0.6)
        for n, c in COLORS.items():
            if n in lmc:
                x, y, z = lmc[n]["voxel"]
                a.plot((x, y)[col], box[2] - 1 - z, "+", color=c, ms=8, mew=1.5)
        a.axis("off")
    ax[0].set_title(f"{pid} ({side}) AP", fontsize=7)
    ax[1].set_title("LAT", fontsize=7)


def contact_sheet(thumbs, path, box, scale, per_row=10):
    rows = max(1, -(-len(thumbs) // per_row))
    fig, ax = plt.subplots(rows, 2 * per_row, figsize=(1.6 * per_row, 2.4 * rows), squeeze=False)
    for a in ax.flat:
        a.axis("off")
    for n, t in enumerate(thumbs):
        r, c = divmod(n, per_row)
        draw(ax[r, 2 * c:2 * c + 2], t, box, scale)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(root, out, patients=None, box=(96, 128, 256), vox=2.0, scale=6.0):
    root, out = Path(root).expanduser(), Path(out).expanduser()
    sides = read_sides(root / "HFValid_Metadata.csv")
    ids = patients or sorted(d.name for d in (root / "Subjects").iterdir() if d.is_dir())
    (out / "qa").mkdir(parents=True, exist_ok=True)

    rows, thumbs, failed = [], [], []
    t0 = time.time()
    for n, pid in enumerate(ids, 1):
        eta = (time.time() - t0) / (n - 1) * (len(ids) - n + 1) if n > 1 else 0
        print(f"[{n}/{len(ids)}  elapsed {time.time() - t0:.0f}s  ETA {eta:.0f}s] ", end="", flush=True)
        try:
            row, thumb = process(root / "Subjects" / pid, pid, sides[pid], out, box, vox, scale)
            rows.append(row)
            thumbs.append(thumb)
        except Exception as e:  # keep going; report at the end
            failed.append((pid, repr(e)))
            print(f"{pid}: FAILED {e!r}")

    fields = ["patient", "side", "ct", "seg", "ap", "lat", "landmarks", "touches_border", "ap_max", "lat_max"]
    fields += [f"{n}_{c}" for n in LANDMARKS for c in "xyz"]
    with open(out / "dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if thumbs:
        contact_sheet(thumbs, out / "qa" / "contact_sheet.png", box, scale)

    print(f"\nDone: {len(rows)} ok, {len(failed)} failed")
    for pid, e in failed:
        print(f"  FAILED {pid}: {e}")
    if rows:
        mx = np.array([max(r["ap_max"], r["lat_max"]) for r in rows])
        print(f"DRR L max per patient: min {mx.min():.2f}, median {np.median(mx):.2f}, max {mx.max():.2f}"
              f"  (PNG scale = {scale})")
        if mx.max() > scale:
            print(f"  WARNING: {(mx > scale).sum()} patients clipped; rerun with --scale >= {mx.max():.1f}")
        border = [r["patient"] for r in rows if r["touches_border"]]
        if border:
            print(f"  WARNING: femur touches box border in {border}")
    return rows, failed


def make_synthetic(root, pid, side):
    """Toy femur (shaft + medial spherical head) as CT + LPS STL + landmarks, in HFValid layout.
    The right case is the exact x-mirror of the left one (patient's left = +x in LPS)."""
    s = 1.0 if side == "L" else -1.0
    size, spacing = (100, 100, 110), (2.0, 2.0, 4.0)
    origin = (1.0 if side == "L" else -199.0, -99.0, -900.0)
    z, y, x = np.meshgrid(*(o + sp * np.arange(n) for o, sp, n in zip(origin[::-1], spacing[::-1], size[::-1])),
                          indexing="ij")
    shaft_x, head, knee = 100 * s, np.array([60 * s, 0, -520]), np.array([100 * s, 0, -840])
    bone = (((x - shaft_x) ** 2 + y ** 2 < 15 ** 2) & (z > -860) & (z < -540)) | \
           ((x - head[0]) ** 2 + y ** 2 + (z - head[2]) ** 2 < 24 ** 2)

    d = root / "Subjects" / pid
    d.mkdir(parents=True)
    img = sitk.GetImageFromArray(np.where(bone, 1000, 0).astype(np.int16))
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(d / f"{pid}.nrrd"))

    vimg = vtk.vtkImageData()
    vimg.SetDimensions(*size)
    vimg.SetSpacing(spacing)
    vimg.SetOrigin(origin)
    vimg.GetPointData().SetScalars(numpy_support.numpy_to_vtk(bone.astype(np.uint8).ravel(), deep=True))
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vimg)
    mc.SetValue(0, 1)
    w = vtk.vtkSTLWriter()
    w.SetFileName(str(d / f"{pid}.stl"))
    w.SetInputConnection(mc.GetOutputPort())
    w.SetFileTypeToBinary()
    w.Write()

    with open(d / f"{pid}_palp.inp", "w") as f:
        for n, p in (("femurHead", head), ("kneeCenter", knee)):
            for c, v in zip("XYZ", p):
                f.write(f"{n}{c} = {v}\n")


def selftest():
    tmp = Path(tempfile.mkdtemp())
    for pid, side in (("SynL", "L"), ("SynR", "R")):
        make_synthetic(tmp, pid, side)
    (tmp / "HFValid_Metadata.csv").write_text("PatientID;Side\nSynL;L\nSynR;R\n")
    rows, failed = run(tmp, tmp / "out")
    assert not failed, failed

    segs = {}
    for pid in ("SynL", "SynR"):
        po = tmp / "out" / pid
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(po / "seg.nii.gz"))).astype(bool)
        lmc = json.load(open(po / "landmarks.json"))["landmarks"]
        h, k = np.array(lmc["femurHead"]["voxel"]), np.array(lmc["kneeCenter"]["voxel"])
        for n, v in (("femurHead", h), ("kneeCenter", k)):
            x, y, z = np.round(v).astype(int)
            assert seg[z, y, x], f"{pid}: {n} not inside mask"
        assert h[0] < k[0], f"{pid}: head not medial (-x) in canonical space -> mirroring wrong"
        ap = sitk.GetArrayFromImage(sitk.ReadImage(str(po / "ap.png"))).astype(float)
        bz = ap.shape[0]
        col, row = int(round(h[0])), bz - 1 - int(round(h[2]))
        assert row < bz // 2 and ap[row, col] > ap[bz - 1 - row, col], f"{pid}: hip not at top of AP"
        segs[pid] = seg
    dice = 2 * (segs["SynL"] & segs["SynR"]).sum() / (segs["SynL"].sum() + segs["SynR"].sum())
    assert dice > 0.95, f"left/right canonical masks differ (Dice {dice:.3f})"
    print(f"\nselftest OK (left vs mirrored right Dice {dice:.3f}). Outputs: {tmp / 'out'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", help="HFValid_Collection_v3 folder")
    p.add_argument("out", nargs="?", default="preprocessed", help="output folder (default: preprocessed)")
    p.add_argument("--patients", nargs="*", help="e.g. Pat001 Pat002 (default: all)")
    p.add_argument("--box", type=int, nargs=3, default=[96, 128, 256], metavar=("X", "Y", "Z"), help="box size in voxels")
    p.add_argument("--vox", type=float, default=2.0, help="voxel size in mm")
    p.add_argument("--scale", type=float, default=6.0, help="fixed DRR scale: PNG white = this L value")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.root:
        p.error("root is required (or use --selftest)")
    run(a.root, a.out, a.patients, a.box, a.vox, a.scale)


if __name__ == "__main__":
    main()
