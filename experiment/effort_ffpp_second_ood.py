"""effort_ffpp_second_ood.py -- third transfer target from saved checkpoints.

v1.0 2026-07-12 (all-nighter). The WIFS paper's Celeb-DF deficit (leaky
arm lower at all five seeds) currently stands alone as the cross-task
transfer. The ten per-seed CLIP-arm checkpoints are on disk, so FF++
costs scoring only, no retraining: 10,000 balanced ffpp_v6_crops frames
(5,000 live / 5,000 manipulated), scored by every arm at every seed.
Same analysis as Celeb-DF: video-level AUC (strip the _fNNNNN suffix,
average frames per video), per-seed deltas, sign count, clean-vs-clean
placebo pairs. If the deficit replicates here, the "Celeb-DF quirk"
objection dies.

Outputs:
  reports/killer/_ffpp_inputs.json          the frame list
  reports/killer/_ffpp_{arm}_s{seed}.json   per-sample P(attack)
  reports/killer/ffpp_second_ood_study.json aggregate
"""
import io, json, os, re, subprocess, sys, time
from _notifications import ntfy as _notify

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(REPO, "reports", "killer")
TRAIN_DIR = os.path.join(REPO, "third_party", "Effort-AIGI-Detection-main",
                         "DeepfakeBench", "training")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
FFPP = r"E:\pad_datasets\ffpp_v6_crops"
SEEDS = ["42", "43", "44", "45", "46"]


def ntfy(title, msg, tags="brain"):
    _notify(title, msg, tags=tags)


def ckpt_path(arm, seed):
    base = f"ckpt_{'naive' if arm == 'naive' else 'l0'}"
    return os.path.join(K, f"{base}.pth" if seed == "42" else f"{base}_s{seed}.pth")


def video_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_f\d+$", "", stem)


def video_auc(scores):
    agg = {}
    for p, s in scores.items():
        v = video_id(p)
        agg.setdefault(v, {"y": 0 if "\\live\\" in p else 1, "s": []})["s"].append(s)
    ys = [d["y"] for d in agg.values()]
    ss = [float(np.mean(d["s"])) for d in agg.values()]
    return float(roc_auc_score(ys, ss)), len(agg)


def frame_auc(scores):
    ys = [0 if "\\live\\" in p else 1 for p in scores]
    return float(roc_auc_score(ys, list(scores.values())))


def main():
    paths = []
    for sub in ("live", "spoof"):
        d = os.path.join(FFPP, sub)
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.lower().endswith(".jpg")]
    inputs = os.path.join(K, "_ffpp_inputs.json")
    with io.open(inputs, "w", encoding="utf8") as f:
        json.dump(paths, f)
    n_vid = len(set(video_id(p) for p in paths))
    print(f"[OK] FF++ pool: {len(paths)} frames, {n_vid} videos", flush=True)
    ntfy("FF++ second-OOD scoring STARTED",
         f"{len(paths)} frames / {n_vid} videos x 10 checkpoints", tags="rocket")

    for seed in SEEDS:
        for arm in ("naive", "l0"):
            out = os.path.join(K, f"_ffpp_{arm}_s{seed}.json")
            if os.path.isfile(out):
                print(f"[--] {out} exists, skip", flush=True)
                continue
            ck = ckpt_path(arm, seed)
            if not os.path.isfile(ck):
                print(f"[ERR] missing checkpoint {ck}", flush=True)
                return 1
            t0 = time.time()
            log = os.path.join(K, f"_ffpp_{arm}_s{seed}.log")
            with open(log, "w") as lf:
                rc = subprocess.run(
                    [PY, "_dodo_effort_score.py", "--weights", ck,
                     "--inputs", inputs, "--out", out, "--device", "cuda"],
                    cwd=TRAIN_DIR, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not os.path.isfile(out):
                print(f"[ERR] scorer rc={rc} for {arm} s{seed}, see {log}", flush=True)
                ntfy("FF++ scoring FAILED", f"{arm} s{seed} rc={rc}", tags="x")
                return 1
            mins = (time.time() - t0) / 60
            sc = json.load(open(out))
            va, nv = video_auc(sc)
            print(f"[OK] ffpp {arm} s{seed}: video AUC {va:.4f} "
                  f"({nv} videos, {mins:.1f} min)", flush=True)
            ntfy(f"ffpp {arm} s{seed} scored",
                 f"video AUC {va:.4f} ({mins:.1f} min)", tags="white_check_mark")

    res = {a: {} for a in ("naive", "l0")}
    fres = {a: {} for a in ("naive", "l0")}
    for seed in SEEDS:
        for arm in ("naive", "l0"):
            sc = json.load(open(os.path.join(K, f"_ffpp_{arm}_s{seed}.json")))
            res[arm][seed], _ = video_auc(sc)
            fres[arm][seed] = frame_auc(sc)
    deltas = {s: res["naive"][s] - res["l0"][s] for s in SEEDS}
    dv = list(deltas.values())
    placebo = {}
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1:]:
            placebo[f"{s1}v{s2}"] = res["l0"][s1] - res["l0"][s2]
    summary = {
        "design": "third transfer target, scored from the saved per-seed "
                  "checkpoints of the controlled A/B; video-level AUC",
        "video_auc_naive": res["naive"], "video_auc_l0": res["l0"],
        "frame_auc_naive": fres["naive"], "frame_auc_l0": fres["l0"],
        "delta_per_seed": deltas,
        "delta_mean": float(np.mean(dv)), "delta_sd": float(np.std(dv, ddof=1)),
        "seeds_negative": int(sum(1 for d in dv if d < 0)),
        "placebo_clean_vs_clean": placebo,
        "placebo_max_abs": float(max(abs(v) for v in placebo.values())),
    }
    out = os.path.join(K, "ffpp_second_ood_study.json")
    with io.open(out, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=1)
    print(f"[OK] SUMMARY: delta mean {summary['delta_mean']:+.4f} "
          f"sd {summary['delta_sd']:.4f}, negative {summary['seeds_negative']}/5, "
          f"placebo max {summary['placebo_max_abs']:.4f} -> {out}", flush=True)
    ntfy("FF++ second-OOD COMPLETE",
         f"delta mean {summary['delta_mean']:+.4f} sd {summary['delta_sd']:.4f}, "
         f"negative {summary['seeds_negative']}/5, "
         f"placebo max {summary['placebo_max_abs']:.4f}", tags="tada")
    return 0


if __name__ == "__main__":
    sys.exit(main())

