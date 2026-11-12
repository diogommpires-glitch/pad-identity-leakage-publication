"""_dodo_effort_train.py -- isolated Effort trainer (run with cwd=this dir).

v1.0 2026-06-21. LoRA-light retrain of Effort (CLIP-L14 backbone FROZEN; only
the SVD residual + 2-class head train) on a flat (path,label) list. Standalone
subprocess so DeepfakeBench's generic utils/detectors packages never collide
with the main DodoVisionAI repo.

Input  : --train-list <json>  = [{"path": "...", "label": 0|1}, ...]
Output : --out <ckpt.pth>     = model.state_dict()
Run    : python _dodo_effort_train.py --train-list tr.json --out ck.pth --epochs 8
         (cwd MUST be DeepfakeBench/training). --preflight = 1 mini-batch sanity.
"""
import argparse, contextlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ at front

import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class CropDataset(Dataset):
    def __init__(self, items):
        self.items = items  # list of {"path","label"}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(it["path"])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)  # fail-safe; rare
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        t = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(t).permute(2, 0, 1).contiguous()
        return t, int(it["label"])


def _build(cfg_path, device):
    from detectors.effort_detector import EffortDetector  # import here (workers stay light)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = EffortDetector(cfg).to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=os.path.join("config", "detector", "effort.yaml"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preflight", action="store_true", help="1 mini-batch sanity then exit")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if not os.path.isfile(a.train_list):
        print(f"[ERR] train-list missing: {a.train_list}", file=sys.stderr); sys.exit(2)
    items = json.load(open(a.train_list))
    if not items:
        print("[ERR] empty train-list", file=sys.stderr); sys.exit(2)
    print(f"[..] build Effort on {a.device} | train n={len(items)} epochs={a.epochs} lr={a.lr}", file=sys.stderr)

    model = _build(a.cfg, a.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[OK] trainable params: {n_tr:,} (backbone frozen)", file=sys.stderr)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay)
    # v1.1: bf16 autocast on cuda (fp32 CLIP-L14 was ~13s/iter on this GPU).
    use_amp = a.device == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    ds = CropDataset(items)
    if a.preflight:
        xb = torch.stack([ds[i][0] for i in range(min(a.batch, len(ds)))]).to(a.device)
        yb = torch.tensor([ds[i][1] for i in range(min(a.batch, len(ds)))], dtype=torch.long).to(a.device)
        model.train()
        with amp:
            pred = model({"image": xb, "label": yb})
            loss = model.get_losses({"image": xb, "label": yb}, pred)["overall"]
        loss.backward(); opt.step()
        print(f"[OK] preflight loss={float(loss):.4f} (forward+backward+step ok, amp={use_amp})", file=sys.stderr)
        sys.exit(0)

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=False, pin_memory=(a.device == "cuda"))
    model.train()
    for ep in range(a.epochs):
        tot, correct, lsum, nb = 0, 0, 0.0, 0
        t_ep = time.time()
        for xb, yb in dl:
            xb = xb.to(a.device, non_blocking=True)
            yb = yb.to(a.device, non_blocking=True)
            opt.zero_grad()
            with amp:
                pred = model({"image": xb, "label": yb})
                loss = model.get_losses({"image": xb, "label": yb}, pred)["overall"]
            loss.backward(); opt.step()
            lsum += float(loss.detach()); nb += 1
            correct += int((pred["cls"].argmax(1) == yb).sum()); tot += yb.numel()
            if nb % 50 == 0:
                r = nb * a.batch / max(1e-9, time.time() - t_ep)
                print(f"    iter {nb}  loss={lsum/nb:.4f}  {r:.1f} imgs/s", file=sys.stderr, flush=True)
        print(f"    epoch {ep+1}/{a.epochs}  loss={lsum/max(1,nb):.4f}  acc={correct/max(1,tot):.4f}  ({time.time()-t_ep:.0f}s)", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print(f"[OK] saved {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
