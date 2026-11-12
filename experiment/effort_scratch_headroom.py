"""effort_scratch_headroom.py -- the missing controlled-inflation experiment.

v1.0 2026-07-12 (all-nighter). The WIFS paper's controlled A/B used a
frozen CLIP-L14 probe and returned an in-domain saturated null; the
inflation evidence (0.998 vs 0.870) stayed an uncontrolled discovery.
This script closes that gap: a small from-scratch CNN (headroom by
construction) trained on the SAME leaky/clean 12,000-image pools and
scored on the SAME fixed 15,816-image held-out test, 5 seeds per arm.
If the leaky arm beats the clean arm here, inflation is demonstrated
under the controlled design, one moving part, on the configuration
that has room to inflate.

Pools/test load from the published supplementary zip (canonical), so
the runs are byte-tied to what reviewers hold. bf16 autocast on cuda.
Optional authenticated notifications per completed run.

Outputs:
  reports/killer/_scratch_{arm}_s{seed}.json   per-sample P(attack) on the fixed test
  reports/killer/scratch_headroom_study.json   aggregate (AUCs, deltas, placebos)
"""
import argparse, io, json, os, random, sys, time, zipfile
from _notifications import ntfy as _notify

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KILLER = os.path.join(REPO, "reports", "killer")


def ntfy(title, msg, tags="brain"):
    _notify(title, msg, tags=tags)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_splits(archive=None):
    archive = archive or os.environ.get("DODO_SUPPLEMENTARY_ZIP")
    if not archive:
        raise ValueError("Set --supplementary-zip or DODO_SUPPLEMENTARY_ZIP to the supplementary archive")
    with zipfile.ZipFile(archive) as z:
        leaky = json.loads(z.read("splits/train_leaky_arm.json"))
        clean = json.loads(z.read("splits/train_clean_arm_L0.json"))
        test = json.loads(z.read("splits/test_fixed_heldout.json"))
    test_recs = [{"path": p, "label": 0 if "\\live\\" in p else 1} for p in test]
    return {"leaky": leaky, "clean": clean}, test_recs


class CropSet(Dataset):
    def __init__(self, recs, img):
        self.recs, self.img = recs, img

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB").resize((self.img, self.img), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return (x - 0.5) / 0.5, int(r["label"]), i


class ScratchCNN(nn.Module):
    """Compact from-scratch detector, parameter scale near the CLIP arms'
    198,754-param residual so the contrast is regime, not budget."""

    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 128]
        blocks = []
        for a, b in zip(ch, ch[1:]):
            blocks += [nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(ch[-1], 2)

    def forward(self, x):
        f = self.features(x).mean(dim=(2, 3))
        return self.head(f)


def run_arm(arm, recs, test_recs, seed, a):
    t0 = time.time()
    seed_all(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScratchCNN().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    tl = DataLoader(CropSet(recs, a.img), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    model.train()
    for ep in range(a.epochs):
        tot = hits = seen = 0.0
        for x, y, _ in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(x)
                loss = ce(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            hits += (out.argmax(1) == y).sum().item()
            seen += len(y)
        print(f"[OK] {arm} s{seed} ep{ep+1}/{a.epochs} loss {tot/seen:.4f} "
              f"train-acc {hits/seen:.4f}", flush=True)

    model.eval()
    sl = DataLoader(CropSet(test_recs, a.img), batch_size=a.batch * 2,
                    shuffle=False, num_workers=a.workers, pin_memory=True)
    scores = np.zeros(len(test_recs))
    ys = np.array([r["label"] for r in test_recs])
    with torch.no_grad():
        for x, _, idx in sl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                p = torch.softmax(model(x.to(dev, non_blocking=True)), 1)[:, 1]
            scores[idx.numpy()] = p.float().cpu().numpy()
    auc = float(roc_auc_score(ys, scores))
    out = os.path.join(KILLER, f"_scratch_{arm}_s{seed}.json")
    with io.open(out, "w", encoding="utf8") as f:
        json.dump({r["path"]: float(s) for r, s in zip(test_recs, scores)}, f)
    mins = (time.time() - t0) / 60
    print(f"[OK] {arm} s{seed}: held-out AUC {auc:.4f} ({mins:.1f} min, "
          f"{n_par:,} params) -> {out}", flush=True)
    ntfy(f"scratch {arm} s{seed} done",
         f"held-out AUC {auc:.4f} ({mins:.1f} min)", tags="white_check_mark")
    return auc, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=112)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--supplementary-zip")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    pools, test_recs = load_splits(a.supplementary_zip)
    if a.smoke:
        for k in pools:
            pools[k] = pools[k][:512]
        test_recs = test_recs[:1024]
        a.epochs, seeds = 1, seeds[:1]
        print("[--] SMOKE mode: 512 train / 1024 test / 1 epoch / 1 seed", flush=True)
    for k, v in pools.items():
        live = sum(1 for r in v if r["label"] == 0)
        print(f"[OK] pool {k}: {len(v)} imgs ({live} live / {len(v)-live} spoof)", flush=True)
    print(f"[OK] fixed test: {len(test_recs)} imgs", flush=True)

    ntfy("scratch headroom study STARTED",
         f"{len(seeds)} seeds x 2 arms, {a.epochs} epochs, img {a.img}", tags="rocket")
    res = {"leaky": {}, "clean": {}, "params": None}
    for seed in seeds:
        for arm in ("leaky", "clean"):
            auc, n_par = run_arm(arm, pools[arm], test_recs, seed, a)
            res[arm][str(seed)] = auc
            res["params"] = n_par

    if not a.smoke and len(seeds) > 1:
        deltas = {s: res["leaky"][str(s)] - res["clean"][str(s)] for s in seeds}
        dv = list(deltas.values())
        placebo = {}
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1:]:
                placebo[f"{s1}v{s2}"] = res["clean"][str(s1)] - res["clean"][str(s2)]
        summary = {
            "design": "same pools/test as the CLIP A/B (supplementary v4), "
                      "from-scratch CNN, headroom regime",
            "params": res["params"], "epochs": a.epochs, "img": a.img,
            "auc_leaky": res["leaky"], "auc_clean": res["clean"],
            "delta_per_seed": {str(k): v for k, v in deltas.items()},
            "delta_mean": float(np.mean(dv)), "delta_sd": float(np.std(dv, ddof=1)),
            "seeds_leaky_higher": int(sum(1 for d in dv if d > 0)),
            "placebo_clean_vs_clean": placebo,
            "placebo_max_abs": float(max(abs(v) for v in placebo.values())),
        }
        out = os.path.join(KILLER, "scratch_headroom_study.json")
        with io.open(out, "w", encoding="utf8") as f:
            json.dump(summary, f, indent=1)
        print(f"[OK] SUMMARY: delta mean {summary['delta_mean']:+.4f} "
              f"sd {summary['delta_sd']:.4f}, leaky higher {summary['seeds_leaky_higher']}/{len(seeds)}, "
              f"placebo max {summary['placebo_max_abs']:.4f} -> {out}", flush=True)
        ntfy("scratch headroom study COMPLETE",
             f"delta mean {summary['delta_mean']:+.4f} sd {summary['delta_sd']:.4f}, "
             f"leaky higher {summary['seeds_leaky_higher']}/{len(seeds)}, "
             f"placebo max {summary['placebo_max_abs']:.4f}", tags="tada")
    return 0


if __name__ == "__main__":
    sys.exit(main())

