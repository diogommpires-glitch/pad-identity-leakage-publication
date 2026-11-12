"""effort_seed_robustness.py -- the two paper-strengthening runs, end to end.

v1.0 2026-07-07. Stages (each ntfys the phone on start + result):
  A. SECOND OOD TARGET: score the two seed-42 arms on the Celeb-DF aligned
     crops (video-level AUC, video-bootstrap CI on the delta).
  B. SEED ROBUSTNESS: retrain BOTH arms at seeds 43 and 44 on the SAME
     fixed pools (only the training seed moves).
  C. Score every new checkpoint on the fixed in-domain test and OOD Replay.
  D. ANALYSIS: per-seed OOD delta + subject-bootstrap CI; mean+-sd across
     seeds 42/43/44; clean-vs-clean placebo delta (the noise floor).
Outputs: reports/killer/second_ood_celebdf.json + seed_robustness.json.
Log: reports/killer/seed_run.log ([STAGE] lines for the monitor).
"""
import json, os, re, subprocess, sys, time

from _notifications import ntfy as _notify

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(REPO, "reports", "killer")
TRAIN_DIR = os.path.join(REPO, "third_party", "Effort-AIGI-Detection-main", "DeepfakeBench", "training")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
LOG = os.path.join(K, "seed_run.log")
CDF = r"E:\pad_datasets\celebdf_aligned_test"

import numpy as np
from sklearn.metrics import roc_auc_score


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
        ntfy("SEED RUN FAILED", f"{label} rc={rc}")
        raise RuntimeError(f"{label} failed rc={rc}")


def score(weights, inputs_json, out_json, label):
    run([PY, "_dodo_effort_score.py", "--weights", weights,
         "--inputs", inputs_json, "--out", out_json, "--device", "cuda"], label)
    return json.load(open(out_json))


def subject_bootstrap(y, pa, pb, groups, n=4000, seed=42):
    subs = np.unique(groups)
    gidx = {s: np.where(groups == s)[0] for s in subs}
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n):
        ss = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([gidx[s] for s in ss])
        if len(set(y[idx].tolist())) < 2:
            continue
        ds.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds >= 0).mean())


def main():
    open(LOG, "w").close()
    t0 = time.time()
    ntfy("SEED+2ndOOD RUN START", "A: 2nd OOD (Celeb-DF). B: retrain seeds 43/44. C: rescore. D: analysis. ETA ~60 min.")

    # ---------- STAGE A: second OOD target ----------
    log("[STAGE] A second-OOD Celeb-DF")
    cdf_items = []
    for sub, lab in (("real", 0), ("fake", 1)):
        d = os.path.join(CDF, sub)
        for fn in os.listdir(d):
            if fn.lower().endswith(".jpg"):
                stem = re.sub(r"__f\d+$", "", os.path.splitext(fn)[0])
                cdf_items.append((os.path.join(d, fn), stem, lab))
    log(f"[OK] celebdf aligned: {len(cdf_items)} frames, {len({s for _, s, _ in cdf_items})} videos")
    inp = os.path.join(K, "_cdf_inputs.json")
    json.dump([p for p, _, _ in cdf_items], open(inp, "w"))
    res_a = {}
    for arm, ck in (("naive", "ckpt_naive.pth"), ("l0", "ckpt_l0.pth")):
        s = score(os.path.join(K, ck), inp, os.path.join(K, f"_cdf_scores_{arm}.json"),
                  f"A: score {arm} on Celeb-DF")
        from collections import defaultdict
        vp = defaultdict(list); vl = {}
        for p, stem, lab in cdf_items:
            if p in s:
                vp[stem].append(s[p]); vl[stem] = lab
        vids = sorted(vp)
        yv = np.array([vl[v] for v in vids]); pv = np.array([np.mean(vp[v]) for v in vids])
        res_a[arm] = {"video_auc": float(roc_auc_score(yv, pv)), "n_videos": len(vids)}
        log(f"[OK] A {arm}: video-AUC {res_a[arm]['video_auc']:.4f}")
    # paired video bootstrap on the delta
    from collections import defaultdict
    svn = json.load(open(os.path.join(K, "_cdf_scores_naive.json")))
    svl = json.load(open(os.path.join(K, "_cdf_scores_l0.json")))
    vpn = defaultdict(list); vpl = defaultdict(list); vl = {}
    for p, stem, lab in cdf_items:
        if p in svn and p in svl:
            vpn[stem].append(svn[p]); vpl[stem].append(svl[p]); vl[stem] = lab
    vids = sorted(vpn)
    yv = np.array([vl[v] for v in vids])
    pn = np.array([np.mean(vpn[v]) for v in vids]); pl = np.array([np.mean(vpl[v]) for v in vids])
    d_a = float(roc_auc_score(yv, pn) - roc_auc_score(yv, pl))
    lo, hi, ppos = subject_bootstrap(yv, pn, pl, np.array(vids))
    res_a["delta_naive_minus_l0"] = d_a
    res_a["ci95_video"] = [lo, hi]
    res_a["note"] = "cross-attack-family transfer (PAD arms scored on deepfakes); interpret the DELTA, not the absolute level"
    json.dump(res_a, open(os.path.join(K, "second_ood_celebdf.json"), "w"), indent=2)
    log(f"[RESULT] A: delta {d_a:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
    ntfy("Stage A done: 2nd OOD Celeb-DF",
         f"naive {res_a['naive']['video_auc']:.3f} vs L0 {res_a['l0']['video_auc']:.3f}, delta {d_a:+.4f} CI[{lo:+.3f},{hi:+.3f}]")

    # ---------- STAGE B: retrain at seeds 43, 44 ----------
    for seed in (43, 44):
        for arm in ("naive", "l0"):
            log(f"[STAGE] B train {arm} seed {seed}")
            ntfy(f"Stage B: training {arm} seed {seed}", "3 epochs, 12k imgs, bf16")
            run([PY, "_dodo_effort_train.py",
                 "--train-list", os.path.join(K, f"leak_train_{arm}.json"),
                 "--out", os.path.join(K, f"ckpt_{arm}_s{seed}.pth"),
                 "--epochs", "3", "--lr", "0.0002", "--batch", "64",
                 "--workers", "8", "--device", "cuda", "--seed", str(seed)],
                f"B: train {arm} s{seed}")

    # ---------- STAGE C: score all new ckpts ----------
    test_in = os.path.join(K, "leak_test_paths.json")
    # replay inputs
    recs = [json.loads(l) for l in open(os.path.join(REPO, "reports", "meta_train_records_replay_v5.jsonl"), encoding="utf-8") if l.strip()]
    rp_paths = [r["path"] for r in recs if r.get("path")]
    rp_in = os.path.join(K, "_replay_inputs.json")
    json.dump(rp_paths, open(rp_in, "w"))
    for seed in (43, 44):
        for arm in ("naive", "l0"):
            ck = os.path.join(K, f"ckpt_{arm}_s{seed}.pth")
            log(f"[STAGE] C score {arm} s{seed}")
            score(ck, test_in, os.path.join(K, f"_indom_{arm}_s{seed}.json"), f"C: in-domain {arm} s{seed}")
            score(ck, rp_in, os.path.join(K, f"_replay_{arm}_s{seed}.json"), f"C: replay {arm} s{seed}")
            ntfy(f"Stage C: scored {arm} s{seed}", "in-domain + replay done")

    # ---------- STAGE D: analysis ----------
    log("[STAGE] D analysis")
    pat = re.compile(r"client(\d+)", re.I)
    y_r, g_r, path_r = [], [], []
    for r in recs:
        p = r.get("path")
        if p:
            y_r.append(1 if str(r.get("label")).lower() in ("spoof", "attack", "1", "true") else 0)
            g_r.append(pat.search(p).group(0).lower()); path_r.append(p)
    y_r = np.array(y_r); g_r = np.array(g_r)

    def replay_vec(fjson):
        s = json.load(open(fjson))
        mask = [i for i, p in enumerate(path_r) if p in s]
        return np.array([s[path_r[i]] for i in mask]), y_r[mask], g_r[mask]

    def indom_auc(fjson):
        s = json.load(open(fjson))
        paths = json.load(open(test_in))
        SP = os.sep + "spoof" + os.sep
        yy = []; pp = []
        for p in paths:
            if p in s:
                yy.append(1 if SP in p else 0); pp.append(s[p])
        return float(roc_auc_score(np.array(yy), np.array(pp)))

    out = {"seeds": {}, "placebo": {}}
    # seed 42 baselines from the original run
    out["seeds"]["42"] = {"ood_delta": -0.0278, "ood_ci": [-0.0417, -0.0140],
                           "indom_naive": 0.99997, "indom_l0": 0.99998, "source": "original run"}
    prev_clean = {}
    for seed in (43, 44):
        pn_, yn_, gn_ = replay_vec(os.path.join(K, f"_replay_naive_s{seed}.json"))
        pl_, yl_, gl_ = replay_vec(os.path.join(K, f"_replay_l0_s{seed}.json"))
        assert len(pn_) == len(pl_) and (yn_ == yl_).all()
        d = float(roc_auc_score(yn_, pn_) - roc_auc_score(yl_, pl_))
        lo, hi, ppos = subject_bootstrap(yn_, pn_, pl_, gn_)
        ain = indom_auc(os.path.join(K, f"_indom_naive_s{seed}.json"))
        ail = indom_auc(os.path.join(K, f"_indom_l0_s{seed}.json"))
        out["seeds"][str(seed)] = {"ood_delta": d, "ood_ci": [lo, hi], "p_ge_0": ppos,
                                    "ood_auc_naive": float(roc_auc_score(yn_, pn_)),
                                    "ood_auc_l0": float(roc_auc_score(yl_, pl_)),
                                    "indom_naive": ain, "indom_l0": ail}
        prev_clean[seed] = (pl_, yl_, gl_)
        log(f"[RESULT] D seed {seed}: OOD delta {d:+.4f} CI[{lo:+.4f},{hi:+.4f}] | in-domain {ain:.5f}/{ail:.5f}")
        ntfy(f"Seed {seed} result", f"OOD delta {d:+.4f} CI[{lo:+.3f},{hi:+.3f}]")

    # placebo: clean@43 vs clean@42 and clean@44 vs clean@43 (noise floor)
    p42 = json.load(open(os.path.join(K, "_replay_scores_l0.json")))
    mask = [i for i, p in enumerate(path_r) if p in p42]
    p42v = np.array([p42[path_r[i]] for i in mask]); y42 = y_r[mask]; g42 = g_r[mask]
    for seed in (43, 44):
        plv, ylv, glv = prev_clean[seed]
        n = min(len(p42v), len(plv))
        d = float(roc_auc_score(y42[:n], plv[:n]) - roc_auc_score(y42[:n], p42v[:n]))
        lo, hi, _ = subject_bootstrap(y42[:n], plv[:n], p42v[:n], g42[:n])
        out["placebo"][f"clean_s{seed}_vs_clean_s42"] = {"delta": d, "ci": [lo, hi]}
        log(f"[RESULT] D placebo clean s{seed} vs s42: {d:+.4f} CI[{lo:+.4f},{hi:+.4f}]")

    deltas = [out["seeds"][s]["ood_delta"] for s in ("42", "43", "44")]
    out["summary"] = {"ood_delta_mean": float(np.mean(deltas)), "ood_delta_sd": float(np.std(deltas, ddof=1)),
                       "all_negative": bool(all(d < 0 for d in deltas)),
                       "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(os.path.join(K, "seed_robustness.json"), "w"), indent=2)
    log(f"[STAGE] DONE mean {out['summary']['ood_delta_mean']:+.4f} sd {out['summary']['ood_delta_sd']:.4f} all_neg={out['summary']['all_negative']} ({out['summary']['minutes']} min)")
    ntfy("SEED+2ndOOD COMPLETE",
         f"OOD delta mean {out['summary']['ood_delta_mean']:+.4f} sd {out['summary']['ood_delta_sd']:.4f}, all seeds negative: {out['summary']['all_negative']}. Placebo + 2nd-OOD in JSONs.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERR] {e}")
        ntfy("SEED RUN CRASHED", str(e)[:180])
        raise

