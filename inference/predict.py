"""3D femur mask, hip/knee centres and mechanical axis from an AP and a lateral X-ray.

The X-rays must be in the format of this project's DRRs: AP 256x96 and LAT 256x128 pixels at 2 mm,
hip at the top, femur roughly centred (e.g. preprocessed/Pat001/ap.png and lat.png).
Weights are downloaded from Hugging Face (ayyp/biplanar-xray-to-3d-femur) on first use; the 5 fold models are averaged.

  python inference/predict.py --ap ap.png --lat lat.png --side L --out prediction/
"""
import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from femuraxis.common import SHAPE, VOX, load_models, predict, read
from femuraxis.landmarks import KNEE_OFFSET_HFVALID, angle, fit_head, fit_knee

WEIGHTS_URL = "https://huggingface.co/ayyp/biplanar-xray-to-3d-femur/resolve/main"


def download_weights(model, folder, n_folds=5):
    folder.mkdir(parents=True, exist_ok=True)
    for f in range(n_folds):
        path = folder / f"{model}_fold{f}.pt"
        if not path.exists():
            print(f"downloading {path.name}")
            urllib.request.urlretrieve(f"{WEIGHTS_URL}/{path.name}", path)


def load_xray(path, shape):
    im = read(path).astype(np.float32)
    if im.ndim == 3:
        im = im[..., 0]
    if im.shape != shape:
        raise ValueError(f"{path}: expected {shape[0]}x{shape[1]} pixels at 2 mm, got {im.shape[0]}x{im.shape[1]}")
    return (im / im.max())[::-1].copy()  # row index = z, hip at the last row


def preview(ap, lat, mask, hip, knee, path):
    fig, axs = plt.subplots(1, 2, figsize=(6, 6))
    for a, img, sil, (c, r) in ((axs[0], ap, mask.any(1), (0, 2)), (axs[1], lat, mask.any(2), (1, 2))):
        a.imshow(img, cmap="gray", origin="lower")
        a.contour(sil, levels=[0.5], colors="#eb6834", linewidths=0.8)
        a.plot([hip[c], knee[c]], [hip[r], knee[r]], c="#2a78d6", lw=1.5)
        a.plot(*hip[[c, r]], "+", c="lime", ms=10, mew=2)
        a.plot(*knee[[c, r]], "+", c="magenta", ms=10, mew=2)
        a.axis("off")
    axs[0].set_title("AP")
    axs[1].set_title("LAT")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", required=True)
    p.add_argument("--lat", required=True)
    p.add_argument("--side", choices=["L", "R"], default="L", help="which leg; right legs are mirrored to look left")
    p.add_argument("--model", default="attnC", choices=["attnC", "swinC"])
    p.add_argument("--weights", default="weights")
    p.add_argument("--out", default="prediction")
    a = p.parse_args()

    ap, lat = load_xray(a.ap, SHAPE[::2]), load_xray(a.lat, SHAPE[:2])
    if a.side == "R":
        ap = ap[:, ::-1].copy()

    download_weights(a.model, Path(a.weights))
    mask = predict(load_models(a.model, a.weights), ap, lat)
    hip, knee = fit_head(mask), fit_knee(mask) + KNEE_OFFSET_HFVALID
    if a.side == "R":
        mask, ap = mask[:, :, ::-1], ap[:, ::-1]
        hip[0], knee[0] = SHAPE[2] - 1 - hip[0], SHAPE[2] - 1 - knee[0]

    axis = hip - knee
    result = dict(hip_vox=hip.round(2).tolist(), knee_vox=knee.round(2).tolist(),
                  axis_length_mm=round(float(np.linalg.norm(axis) * VOX), 1),
                  ap_tilt_deg=round(angle(axis[[0, 2]], np.array([0, 1])), 2),
                  lat_tilt_deg=round(angle(axis[[1, 2]], np.array([0, 1])), 2))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing([VOX] * 3)
    sitk.WriteImage(img, str(out / "mask.nii.gz"), useCompression=True)
    json.dump(result, open(out / "result.json", "w"), indent=2)
    preview(ap, lat, mask, hip, knee, out / "preview.png")

    print(f"hip centre   {result['hip_vox']} (voxels of {VOX:g} mm)")
    print(f"knee centre  {result['knee_vox']}")
    print(f"axis length  {result['axis_length_mm']} mm")
    print(f"axis tilt from vertical: {result['ap_tilt_deg']} deg in the AP view, {result['lat_tilt_deg']} deg in the lateral view")
    print(f"saved mask.nii.gz, result.json, preview.png in {out}")


if __name__ == "__main__":
    main()
