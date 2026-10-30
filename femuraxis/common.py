import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from monai.networks.nets import AttentionUnet, SwinUNETR
from scipy import ndimage

SHAPE = (256, 128, 96)  # (z, y, x) of the canonical box
VOX = 2.0
DRR_SCALE = 6.0

device = "cuda" if torch.cuda.is_available() else "cpu"
amp = device == "cuda"

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
ARCHS = {"AttentionUnet": AttentionUnet, "SwinUNETR": SwinUNETR}


def load_config(name_or_path):
    """A model name (attnC) is looked up in configs/, anything else is read as a path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIGS / f"{name_or_path}.yaml"
    return yaml.safe_load(open(path))


def build_model(cfg):
    kw = dict(cfg["model"])
    return ARCHS[kw.pop("arch")](spatial_dims=3, in_channels=2, out_channels=1, **kw).to(device)


def read(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def load_split(path):
    folds = json.load(open(path))
    fold_of = {p: k for k, f in enumerate(folds) for p in f}
    return folds, fold_of


class HFValid:
    """Preprocessed HFValid cases. DRRs are returned with row index = z (hip at the last row)."""

    def __init__(self, root, ids):
        self.root = Path(root)
        self.ids = list(ids)
        self.drr = {p: [read(self.root / p / f"{v}.png").astype(np.float32)[::-1] / 65535.0 for v in ("ap", "lat")]
                    for p in self.ids}
        self.seg = {p: read(self.root / p / "seg.nii.gz").astype(bool) for p in self.ids}
        self._ct = {}

    def ct(self, pid):
        if pid not in self._ct:
            self._ct[pid] = read(self.root / pid / "ct.nii.gz").astype(np.int16)
        return self._ct[pid]

    def landmarks(self, pid):
        lm = json.load(open(self.root / pid / "landmarks.json"))["landmarks"]
        return np.array(lm["femurHead"]["voxel"]), np.array(lm["kneeCenter"]["voxel"])


def project(hu):
    """Parallel-beam AP and LAT DRRs of a (z, y, x) HU volume, same formula as preprocess_hfvalid.py."""
    mu = np.clip(0.02 * (1 + hu / 1000.0), 0, None)
    return [np.clip(mu.sum(axis=a) * VOX / DRR_SCALE, 0, 1).astype(np.float32) for a in (1, 2)]


def normalise(im):
    return ((im - im.mean()) / (im.std() + 1e-6)).astype(np.float32)


def fuse(ap, lat):
    """Repeat each DRR along its ray direction and stack: (2, z, y, x)."""
    ap, lat = normalise(ap), normalise(lat)
    return np.stack([np.broadcast_to(ap[:, None, :], SHAPE), np.broadcast_to(lat[:, :, None], SHAPE)]).astype(np.float32)


def drop_small(mask, frac=0.05):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.isin(lab, np.nonzero(sizes >= frac * sizes.max())[0])


def load_models(name, ckpt_dir, n_folds=5):
    nets = []
    for f in range(n_folds):
        net = build_model(load_config(name))
        net.load_state_dict(torch.load(Path(ckpt_dir) / f"{name}_fold{f}.pt", map_location=device))
        nets.append(net.eval())
    return nets


@torch.no_grad()
def predict(nets, ap, lat):
    """Mean probability of the given models, thresholded at 0.5, small blobs removed."""
    x = torch.from_numpy(fuse(ap, lat))[None].to(device)
    with torch.autocast(device_type="cuda", enabled=amp):
        p = torch.stack([torch.sigmoid(n(x)) for n in nets]).mean(0)
    return drop_small((p > 0.5)[0, 0].cpu().numpy())
