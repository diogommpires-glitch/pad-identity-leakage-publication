"""effort_seed_extend.py -- extend the seed study to seeds 45 and 46.

v1.0 2026-07-08. Trains both arms at two more seeds (same frozen pools),
scores Celeb-DF + Replay + in-domain, appends per-seed deltas, and prints
the 5-seed verdict (sign test on the Celeb-DF within-pair delta).
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
SEEDS = (45, 46)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ntfy(title, body):
    _notify(title, body, tags="gear")


def run(cmd, label):
    log(f"[..] {label}")
    with open(LOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=TRAIN_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"[{'OK' if rc == 0 else 'ERR'}] {label} rc={rc}")
    if rc != 0:
        ntfy("SEED EXTEND FAILED", f"{label} rc={rc}")
        raise RuntimeError(label)


def main():
    t0 = time.time()
    ntfy("SEED EXTEND START", "Seeds 45+46: train both arms, score CDF/Replay/in-domain. ~40 min.")

    cdf_items = []
    for sub, lab in (("real", 0), ("fake", 1)):
        d = os.path.join(CDF, sub)
        for fn in os.listdir(d):
            if fn.lower().endswith(".jpg"):
                stem = re.sub(r"__f\d+$", "", os.path.splitext(fn)[0])
                cdf_items.append((os.path.join(d, fn), stem, lab))
    cdf_in = os.path.join(K, "_cdf_inputs.json")
    json.dump([p for p, _, _ in cdf_items], open(cdf_in, "w"))

    recs = [json.loads(l) for l in open(os.path.join(REPO, "reports", "meta_train_records_replay_v5.jsonl"), encoding="utf-8") if l.strip()]
    rp_in = os.path.join(K, "_replay_inputs.json")
    json.dump([r["path"] for r in recs if r.get("path")], open(rp_in, "w"))
    test_in = os.path.join(K, "leak_test_paths.json")

    for seed in SEEDS:
        for arm in ("naive", "l0"):
            log(f"[STAGE] EXT train {arm} seed {seed}")
            run([PY, "_dodo_effort_train.py",
                 "--train-list", os.path.join(K, f"leak_train_{arm}.json"),
                 "--out", os.path.join(K, f"ckpt_{arm}_s{seed}.pth"),
                 "--epochs", "3", "--lr", "0.0002", "--batch", "64",
                 "--workers", "8", "--device", "cuda", "--seed", str(seed)],
                f"EXT train {arm} s{seed}")
        for arm in ("naive", "l0"):
            ck = os.path.join(K, f"ckpt_{arm}_s{seed}.pth")
            log(f"[STAGE] EXT score {arm} s{seed}")
            run([PY, "_dodo_effort_score.py", "--weights", ck, "--inputs", cdf_in,
                 "--out", os.path.join(K, f"_cdf_{arm}_s{seed}.json"), "--device", "cuda"],
                f"EXT cdf {arm} s{seed}")
            run([PY, "_dodo_effort_score.py", "--weights", ck, "--inputs", rp_in,
                 "--out", os.path.join(K, f"_replay_{arm}_s{seed}.json"), "--device", "cuda"],
                f"EXT replay {arm} s{seed}")
            run([PY, "_dodo_effort_score.py", "--weights", ck, "--inputs", test_in,
                 "--out", os.path.join(K, f"_indom_{arm}_s{seed}.json"), "--device", "cuda"],
                f"EXT indom {arm} s{seed}")
        ntfy(f"Seed {seed} trained+scored", "cdf/replay/in-domain done")

    # ---- analysis across ALL FIVE seeds ----
    log("[STAGE] EXT analysis (5 seeds)")

    def video_delta(seed):
        if seed == 42:
            fn_n, fn_l = "_cdf_scores_naive.json", "_cdf_scores_l0.json"
        else:
            fn_n, fn_l = f"_cdf_naive_s{seed}.json", f"_cdf_l0_s{seed}.json"
        sn = json.load(open(os.path.join(K, fn_n))); sl = json.load(open(os.path.join(K, fn_l)))
        vp_n = defaultdict(list); vp_l = defaultdict(list); vl = {}
        for p, stem, lab in cdf_items:
            if p in sn and p in sl:
                vp_n[stem].append(sn[p]); vp_l[stem].append(sl[p]); vl[stem] = lab
        vids = sorted(vp_n)
        y = np.array([vl[v] for v in vids])
        pn = np.array([np.mean(vp_n[v]) for v in vids]); pl = np.array([np.mean(vp_l[v]) for v in vids])
        return float(roc_auc_score(y, pn) - roc_auc_score(y, pl))

    pat = re.compile(r"client(\d+)", re.I)

    def replay_delta(seed):
        if seed == 42:
            fn_n, fn_l = "_replay_scores_naive.json", "_replay_scores_l0.json"
        else:
            fn_n, fn_l = f"_replay_naive_s{seed}.json", f"_replay_l0_s{seed}.json"
        sn = json.load(open(os.path.join(K, fn_n))); sl = json.load(open(os.path.join(K, fn_l)))
        y = []; pn = []; pl = []
        for r in recs:
            p = r.get("path")
            if p and p in sn and p in sl:
                y.append(1 if str(r.get("label")).lower() in ("spoof", "attack", "1", "true") else 0)
                pn.append(sn[p]); pl.append(sl[p])
        y = np.array(y)
        return float(roc_auc_score(y, np.array(pn)) - roc_auc_score(y, np.array(pl)))

    def indom(seed, arm):
        if seed == 42:
            fn = "scores_naive.json" if arm == "naive" else "scores_l0.json"
        else:
            fn = f"_indom_{arm}_s{seed}.json"
        s = json.load(open(os.path.join(K, fn)))
        paths = json.load(open(test_in)); SP = os.sep + "spoof" + os.sep
        yy = []; pp = []
        for p in paths:
            if p in s:
                yy.append(1 if SP in p else 0); pp.append(s[p])
        return float(roc_auc_score(np.array(yy), np.array(pp)))

    seeds_all = (42, 43, 44, 45, 46)
    out = {"cdf_delta": {}, "replay_delta": {}, "indom": {}}
    for s in seeds_all:
        out["cdf_delta"][str(s)] = video_delta(s)
        out["replay_delta"][str(s)] = replay_delta(s)
        out["indom"][str(s)] = {"naive": indom(s, "naive"), "l0": indom(s, "l0")}
        log(f"[RESULT] EXT seed {s}: CDF {out['cdf_delta'][str(s)]:+.4f} | Replay {out['replay_delta'][str(s)]:+.4f} | indom {out['indom'][str(s)]['naive']:.5f}/{out['indom'][str(s)]['l0']:.5f}")

    cdfs = [out["cdf_delta"][str(s)] for s in seeds_all]
    reps = [out["replay_delta"][str(s)] for s in seeds_all]
    n_neg = sum(1 for d in cdfs if d < 0)
    sign_p = 0.5 ** len(cdfs) if n_neg == len(cdfs) else None  # one-sided all-same-sign
    out["summary"] = {
        "cdf_mean": float(np.mean(cdfs)), "cdf_sd": float(np.std(cdfs, ddof=1)),
        "cdf_all_negative": bool(n_neg == len(cdfs)), "cdf_sign_p_onesided": sign_p,
        "replay_mean": float(np.mean(reps)), "replay_sd": float(np.std(reps, ddof=1)),
        "replay_all_negative": bool(all(d < 0 for d in reps)),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    json.dump(out, open(os.path.join(K, "seed_study_5seeds.json"), "w"), indent=2)
    log(f"[STAGE] EXT DONE cdf mean {out['summary']['cdf_mean']:+.4f} sd {out['summary']['cdf_sd']:.4f} all_neg={out['summary']['cdf_all_negative']} sign_p={sign_p} | replay mean {out['summary']['replay_mean']:+.4f} sd {out['summary']['replay_sd']:.4f}")
    ntfy("5-SEED STUDY COMPLETE",
         f"CDF deltas: {'/'.join(f'{d:+.3f}' for d in cdfs)} (all neg: {out['summary']['cdf_all_negative']}, sign p={sign_p}) | Replay: {'/'.join(f'{d:+.3f}' for d in reps)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERR] {e}")
        ntfy("SEED EXTEND CRASHED", str(e)[:180])
        raise

