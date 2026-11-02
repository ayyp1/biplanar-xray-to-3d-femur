"""Export the fold models to ONNX so they run fast on a CPU.

  python deploy/export_onnx.py --weights weights --out onnx
"""
import argparse
from pathlib import Path

import torch

from femuraxis.common import SHAPE, build_model, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="attnC", choices=["attnC", "swinC"])
    p.add_argument("--weights", default="weights", help="folder with <model>_fold<k>.pt")
    p.add_argument("--out", default="onnx")
    p.add_argument("--folds", default="0,1,2,3,4")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.folds.split(","):
        net = build_model(load_config(a.model)).cpu().eval()
        net.load_state_dict(torch.load(Path(a.weights) / f"{a.model}_fold{f}.pt", map_location="cpu"))
        path = out / f"{a.model}_fold{f}.onnx"
        torch.onnx.export(net, torch.zeros(1, 2, *SHAPE), path, opset_version=17, dynamo=False,
                          input_names=["x"], output_names=["y"])
        print("saved", path)


if __name__ == "__main__":
    main()
