"""Train one model on one or more folds. All hyperparameters come from the config file.
  python training/train.py --config configs/attnC.yaml --data preprocessed --fold 0 1 2 3 4
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from femuraxis.common import SHAPE, HFValid, build_model, device, fuse, load_config, load_split, project


def rotation(rx, ry, rz):
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def random_pose(hu, seg, rng, aug):
    """Rotate CT and mask about the box centre, shift them, and re-project the DRRs."""
    R = rotation(*(rng.uniform(-d, d) for d in aug["rotate_deg"]))
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # (x, y, z) <-> array order (z, y, x)
    M = (P @ R @ P).T  # affine_transform maps output -> input, i.e. the inverse rotation
    c = (np.array(SHAPE) - 1) / 2
    offset = c - M @ (c + rng.uniform(-aug["shift_vox"], aug["shift_vox"], 3))
    hu = ndimage.affine_transform(hu.astype(np.float32), M, offset, order=1, cval=-1024.0)
    seg = ndimage.affine_transform(seg.astype(np.float32), M, offset, order=1, cval=0.0) >= 0.5
    return (*project(hu), seg)


def augment_intensity(im, rng, aug):
    im = np.clip(im, 0, 1) ** rng.uniform(*aug["gamma"])
    im = im * rng.uniform(*aug["contrast"]) + rng.uniform(-aug["brightness"], aug["brightness"])
    if rng.random() < aug["blur_prob"]:
        im = ndimage.gaussian_filter(im, rng.uniform(*aug["blur_sigma"]))
    if rng.random() < aug["noise_prob"]:
        im = im + rng.normal(0, rng.uniform(0, aug["noise_std"]), im.shape)
    return im.astype(np.float32)


class Cases(Dataset):
    def __init__(self, data, ids, aug=None):
        self.data, self.ids, self.aug = data, ids, aug

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        ap, lat = self.data.drr[pid]
        seg = self.data.seg[pid]
        if self.aug:
            rng = np.random.default_rng()
            if rng.random() < self.aug["pose_prob"]:
                ap, lat, seg = random_pose(self.data.ct(pid), seg, rng, self.aug)
            ap, lat = augment_intensity(ap, rng, self.aug), augment_intensity(lat, rng, self.aug)
        return torch.from_numpy(fuse(ap, lat)), torch.from_numpy(seg[None].astype(np.float32))


def train_fold(cfg, data, folds, fold, out):
    name, t = cfg["name"], cfg["train"]
    best_p, state_p, done_p = out / f"{name}_fold{fold}.pt", out / f"{name}_fold{fold}_state.pt", out / f"{name}_fold{fold}.json"
    if done_p.exists():
        print(f"{name} fold {fold}: already done")
        return
    val_ids = folds[fold]
    train_ids = [p for k, f in enumerate(folds) if k != fold for p in f]
    train_dl = DataLoader(Cases(data, train_ids, cfg["augment"]), batch_size=t["batch_size"], shuffle=True,
                          num_workers=t["workers"], pin_memory=True, persistent_workers=t["workers"] > 0)
    val_dl = DataLoader(Cases(data, val_ids), batch_size=1, num_workers=t["workers"], pin_memory=True)

    torch.manual_seed(t["seed"] + fold)
    amp = t["amp"] and device == "cuda"
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = DiceCELoss(sigmoid=True)
    dice = DiceMetric(include_background=True, reduction="mean")

    start, best, bad, hist = 1, 0.0, 0, {"loss": [], "dice": []}
    if state_p.exists():
        s = torch.load(state_p, map_location=device, weights_only=False)
        for obj, key in ((model, "model"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")):
            obj.load_state_dict(s[key])
        start, best, bad, hist = s["epoch"] + 1, s["best"], s["bad"], s["hist"]
        print(f"{name} fold {fold}: resuming at epoch {start}")

    for ep in range(start, t["epochs"] + 1):
        if bad >= t["patience"]:
            break
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        model.eval()
        dice.reset()
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", enabled=amp):
                    dice((torch.sigmoid(model(x)) > 0.5).float(), y)
        vd = dice.aggregate().item()
        hist["loss"].append(total / len(train_dl))
        hist["dice"].append(vd)
        if vd > best:
            best, bad = vd, 0
            torch.save(model.state_dict(), best_p)
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep, best=best, bad=bad, hist=hist), state_p)
        print(f"{name} fold {fold} ep {ep:3d}  loss {hist['loss'][-1]:.4f}  val Dice {vd:.4f}  "
              f"best {best:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    json.dump({"best_val_dice": best, "epochs_run": len(hist["dice"]), "hist": hist}, open(done_p, "w"))
    state_p.unlink(missing_ok=True)
    print(f"{name} fold {fold}: done, best val Dice {best:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="e.g. configs/attnC.yaml")
    p.add_argument("--data", required=True, help="preprocessed folder")
    p.add_argument("--split", default=None, help="default: <data>/split.json")
    p.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="checkpoints")
    a = p.parse_args()

    cfg = load_config(a.config)
    folds, _ = load_split(a.split or Path(a.data) / "split.json")
    data = HFValid(a.data, [pid for f in folds for pid in f])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / f"{cfg['name']}.yaml")  # keep the settings next to the weights
    torch.backends.cudnn.benchmark = True
    for f in a.fold:
        train_fold(cfg, data, folds, f, out)


if __name__ == "__main__":
    main()
