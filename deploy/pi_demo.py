"""Run the ONNX models on an AP + LAT pair and get the hip, knee and axis. I ran this on a Raspberry Pi 4.

  python deploy/pi_demo.py --ap ap.png --lat lat.png --onnx onnx

Add --landmarks landmarks.json to compare with the expert points.
"""
import argparse
import json
import resource
import time

import numpy as np
import onnxruntime as ort

from femuraxis.common import SHAPE, VOX, drop_small, fuse, read
from femuraxis.landmarks import KNEE_OFFSET_HFVALID, angle, fit_head, fit_knee


def load_xray(path):
    im = read(path).astype(np.float32)
    if im.ndim == 3:
        im = im[..., 0]
    return (im / im.max())[::-1].copy()  # row index = z, hip at the last row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", required=True)
    p.add_argument("--lat", required=True)
    p.add_argument("--model", default="attnC")
    p.add_argument("--onnx", default="onnx", help="folder with <model>_fold<k>.onnx")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--landmarks")
    a = p.parse_args()

    ap, lat = load_xray(a.ap), load_xray(a.lat)
    if ap.shape != SHAPE[::2] or lat.shape != SHAPE[:2]:
        raise ValueError(f"expected AP {SHAPE[::2]} and LAT {SHAPE[:2]} pixels, got {ap.shape} and {lat.shape}")
    x = fuse(ap, lat)[None]

    opt = ort.SessionOptions()
    opt.enable_cpu_mem_arena = False
    opt.enable_mem_pattern = False
    folds = a.folds.split(",")
    t0, prob = time.time(), 0
    for f in folds:
        t = time.time()
        s = ort.InferenceSession(f"{a.onnx}/{a.model}_fold{f}.onnx", opt, providers=["CPUExecutionProvider"])
        prob = prob + 1 / (1 + np.exp(-s.run(None, {"x": x})[0][0, 0])) / len(folds)
        del s
        print(f"fold {f}: {time.time() - t:.1f} s")
    mask = drop_small(prob > 0.5)
    t_net = time.time() - t0

    t = time.time()
    hip, knee = fit_head(mask), fit_knee(mask) + KNEE_OFFSET_HFVALID
    t_lm = time.time() - t
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024  # KB on Linux

    print(f"network {t_net:.1f} s, landmarks {t_lm:.1f} s, peak RAM {ram} MB")
    print(f"hip centre  {hip.round(2)} (voxels of {VOX:g} mm)")
    print(f"knee centre {knee.round(2)}")
    if a.landmarks:
        lm = json.load(open(a.landmarks))["landmarks"]
        eh, ek = np.array(lm["femurHead"]["voxel"]), np.array(lm["kneeCenter"]["voxel"])
        print(f"hip error {np.linalg.norm(hip - eh) * VOX:.1f} mm, knee error {np.linalg.norm(knee - ek) * VOX:.1f} mm, "
              f"axis error {angle(hip - knee, eh - ek):.2f} deg")


if __name__ == "__main__":
    main()
