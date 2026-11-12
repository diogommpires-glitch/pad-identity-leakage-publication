"""_dodo_effort_score.py -- standalone Effort scorer (run with cwd=this dir).

v1.0 2026-06-21. Isolated subprocess scorer so DeepfakeBench's generic
`utils`/`detectors` packages never collide with the main DodoVisionAI repo.

Input  : --inputs <json>  = list of image paths (absolute).
Output : --out <json>     = {image_path: p_fake}.
Run    : python _dodo_effort_score.py --weights <ckpt.pth> --inputs in.json --out out.json
         (cwd MUST be the DeepfakeBench/training dir).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ at front

import numpy as np
import cv2
import torch
import yaml

_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD = [0.26862954, 0.26130258, 0.27577711]


def _load_model(weights, cfg_path, device):
    from detectors.effort_detector import EffortDetector
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = EffortDetector(cfg)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[OK] loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})", file=sys.stderr)
    model.eval().to(device)
    return model


def _preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    for c in range(3):
        t[c] = (t[c] - _MEAN[c]) / _STD[c]
    return t


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cfg", default=os.path.join("config", "detector", "effort.yaml"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    if not os.path.isfile(a.weights):
        print(f"[ERR] weights not found: {a.weights}", file=sys.stderr); sys.exit(2)
    with open(a.inputs, "r") as f:
        paths = json.load(f)
    print(f"[..] scoring {len(paths)} images on {a.device}", file=sys.stderr)
    model = _load_model(a.weights, a.cfg, a.device)

    out, buf, bpaths = {}, [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(a.device)
        # v1.1: bf16 autocast (matches the trainer; fp32 was ~4s per 64-batch here)
        import contextlib
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if a.device == "cuda" else contextlib.nullcontext()
        with amp:
            preds = model({"image": x, "label": torch.zeros(len(buf), dtype=torch.long).to(a.device)}, inference=True)
        prob = preds["prob"].detach().float().cpu().numpy().reshape(-1)
        for p, pr in zip(bpaths, prob):
            out[p] = float(pr)

    for i, p in enumerate(paths):
        t = _preprocess(p)
        if t is None:
            print(f"[--] unreadable: {p}", file=sys.stderr); continue
        buf.append(t); bpaths.append(p)
        if len(buf) >= a.batch:
            flush(); buf, bpaths = [], []
        if (i + 1) % 512 == 0:
            print(f"    ...{i+1}/{len(paths)}", file=sys.stderr)
    flush()

    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"[OK] wrote {len(out)} scores -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

