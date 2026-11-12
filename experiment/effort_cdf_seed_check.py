"""effort_cdf_seed_check.py -- is the Celeb-DF second-OOD delta seed-stable?

v1.0 2026-07-08. Scores the seed-43/44 checkpoints (both arms) on the
Celeb-DF aligned crops and computes the per-seed video-level delta with a
video-bootstrap CI, alongside the seed-42 result already on disk. Also the
clean-vs-clean placebo on Celeb-DF. Fail-loud; ntfys each result.
"""
import json, os, re, subprocess, sys, time
from _notifications import ntfy as _notify
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(REPO, "reports", "killer")
TRAIN_DIR = os.path.join(REPO, "third_party", "Effort-AIGI-Detection-main", "DeepfakeBench", "training")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
CDF = r"E:\pad_datasets\celebdf_aligned_test"
LOG = os.path.join(K, "seed_run.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ntfy(title, body):
    _notify(title, body, tags="gear")


def main():
    items = []
    for sub, lab in (("real", 0), ("fake", 1)):
        d = os.path.join(CDF, sub)
        for fn in os.listdir(d):
            if fn.lower().endswith(".jpg"):
                stem = re.sub(r"__f\d+$", "", os.path.splitext(fn)[0])
                items.append((os.path.join(d, fn), stem, lab))
    inp = os.path.join(K, "_cdf_inputs.json")
    json.dump([p for p, _, _ in items], open(inp, "w"))

    def video_vec(score_json):
        s = json.load(open(score_json))
        vp = defaultdict(list); vl = {}
        for p, stem, lab in items:
            if p in s:
                vp[stem].append(s[p]); vl[stem] = lab
        vids = sorted(vp)
        return (np.array([vl[v] for v in vids]),
                np.array([np.mean(vp[v]) for v in vids]), np.array(vids))

    def boot(y, pa, pb, vids, n=4000, seed=42):
        rng = np.random.default_rng(seed); ds = []
        gidx = {v: i for i, v in enumerate(vids)}
        for _ in range(n):
            pick = rng.choice(len(vids), len(vids), replace=True)
            if len(set(y[pick].tolist())) < 2:
                continue
            ds.append(roc_auc_score(y[pick], pa[pick]) - roc_auc_score(y[pick], pb[pick]))
        ds = np.array(ds)
        return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

    out = {"per_seed": {}}
    # seed 42 from stage A files
    y, pn, vids = video_vec(os.path.join(K, "_cdf_scores_naive.json"))
    _, pl, _ = video_vec(os.path.join(K, "_cdf_scores_l0.json"))
    d42 = float(roc_auc_score(y, pn) - roc_auc_score(y, pl))
    lo, hi = boot(y, pn, pl, vids)
    out["per_seed"]["42"] = {"delta": d42, "ci": [lo, hi],
                              "auc_naive": float(roc_auc_score(y, pn)),
                              "auc_l0": float(roc_auc_score(y, pl))}
    log(f"[RESULT] CDF seed 42: delta {d42:+.4f} CI[{lo:+.4f},{hi:+.4f}]")

    clean_vecs = {"42": pl}
    for seed in (43, 44):
        for arm in ("naive", "l0"):
            ck = os.path.join(K, f"ckpt_{arm}_s{seed}.pth")
            outj = os.path.join(K, f"_cdf_{arm}_s{seed}.json")
            log(f"[STAGE] CDF score {arm} s{seed}")
            with open(LOG, "a", encoding="utf-8") as f:
                rc = subprocess.run([PY, "_dodo_effort_score.py", "--weights", ck,
                                     "--inputs", inp, "--out", outj, "--device", "cuda"],
                                    cwd=TRAIN_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                ntfy("CDF seed check FAILED", f"{arm} s{seed} rc={rc}"); sys.exit(2)
        y, pn, vids = video_vec(os.path.join(K, f"_cdf_naive_s{seed}.json"))
        _, pl_, _ = video_vec(os.path.join(K, f"_cdf_l0_s{seed}.json"))
        d = float(roc_auc_score(y, pn) - roc_auc_score(y, pl_))
        lo, hi = boot(y, pn, pl_, vids)
        out["per_seed"][str(seed)] = {"delta": d, "ci": [lo, hi],
                                       "auc_naive": float(roc_auc_score(y, pn)),
                                       "auc_l0": float(roc_auc_score(y, pl_))}
        clean_vecs[str(seed)] = pl_
        log(f"[RESULT] CDF seed {seed}: delta {d:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
        ntfy(f"CDF seed {seed}", f"delta {d:+.4f} CI[{lo:+.3f},{hi:+.3f}]")

    # clean-vs-clean placebo on Celeb-DF
    out["placebo"] = {}
    for seed in ("43", "44"):
        d = float(roc_auc_score(y, clean_vecs[seed]) - roc_auc_score(y, clean_vecs["42"]))
        lo, hi = boot(y, clean_vecs[seed], clean_vecs["42"], vids)
        out["placebo"][f"clean_s{seed}_vs_s42"] = {"delta": d, "ci": [lo, hi]}
        log(f"[RESULT] CDF placebo s{seed} vs s42: {d:+.4f} CI[{lo:+.4f},{hi:+.4f}]")

    deltas = [out["per_seed"][s]["delta"] for s in ("42", "43", "44")]
    out["summary"] = {"mean": float(np.mean(deltas)), "sd": float(np.std(deltas, ddof=1)),
                       "all_negative": bool(all(d < 0 for d in deltas))}
    json.dump(out, open(os.path.join(K, "second_ood_celebdf_seeds.json"), "w"), indent=2)
    log(f"[STAGE] CDF-SEED-CHECK DONE mean {out['summary']['mean']:+.4f} sd {out['summary']['sd']:.4f} all_neg={out['summary']['all_negative']}")
    ntfy("CDF seed check COMPLETE",
         f"deltas 42/43/44: {deltas[0]:+.3f}/{deltas[1]:+.3f}/{deltas[2]:+.3f} | mean {out['summary']['mean']:+.4f} sd {out['summary']['sd']:.4f} | all negative: {out['summary']['all_negative']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERR] {e}")
        ntfy("CDF SEED CHECK CRASHED", str(e)[:180])
        raise

