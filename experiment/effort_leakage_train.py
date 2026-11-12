"""effort_leakage_train.py -- the decisive CelebA-Spoof identity-leakage A/B.

v1.0 2026-06-21. Controlled same-test design:
  - Fix a TEST identity set T (held-out images of T = the test set).
  - NAIVE-train: random pool images INCLUDING other images of T's identities
    (identity OVERLAP with test = leaky), size S.
  - L0-train: pool images EXCLUDING all of T's identities (identity-DISJOINT,
    L=0), same size S (size confound controlled).
  - Train Effort (LoRA-light) on each, score BOTH on the SAME test set T.
  - delta-AUC = AUC(naive) - AUC(L0). POSITIVE => identity overlap inflated the
    score = leakage. Paired-bootstrap CI. Reports L(naive)>0 vs L(L0)=0.

Three isolated subprocess pieces dodge the DeepfakeBench `utils` collision:
  this orchestrator (imports pad) builds splits + computes delta; the trainer
  (_dodo_effort_train.py) and scorer (_dodo_effort_score.py) run in cwd=training.

Run (FULL + shutdown):  python scripts/effort_leakage_train.py --shutdown
Run (SMOKE, no shutdown): python scripts/effort_leakage_train.py --smoke
"""
import argparse, json, os, subprocess, sys, time, traceback
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
from pad.leakage_metric import get_extractor, leakage_metrics

PATCH = r"E:\pad_datasets\patch_depth"
TRAIN_DIR = os.path.join(REPO, "third_party", "Effort-AIGI-Detection-main", "DeepfakeBench", "training")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUTD = os.path.join(REPO, "reports", "killer")
LOG = os.path.join(OUTD, "leakage_run.log")
RESULT = os.path.join(OUTD, "effort_leakage_result.json")
STATUS = os.path.join(OUTD, "leakage_status.json")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(stage, **kw):
    d = {"stage": stage, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    json.dump(d, open(STATUS, "w"), indent=2)


def ntfy(title, body):
    try:
        from pad.notify import ntfy as _n
        _n(title, body, tags="trophy")
    except Exception as e:
        log(f"[--] ntfy skipped: {e}")


def run(cmd, label):
    log(f"[..] {label}: {' '.join(str(c) for c in cmd)}")
    with open(LOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=TRAIN_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"[{'OK' if rc == 0 else 'ERR'}] {label} rc={rc}")
    if rc != 0:
        raise RuntimeError(f"{label} failed rc={rc} (see {LOG})")


def inventory(ex):
    """-> dict id -> {0:[live paths], 1:[spoof paths]}"""
    inv = defaultdict(lambda: {0: [], 1: []})
    for folder, label in (("live", 0), ("spoof", 1)):
        d = os.path.join(PATCH, folder)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[ERR] missing {d}")
        for e in os.scandir(d):
            if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                ident = ex({"path": e.path})
                inv[ident][label].append(e.path)
    return inv


def build_splits(args):
    import random
    rng = random.Random(args.seed)
    ex = get_extractor("celeba")
    log("[..] scanning patch_depth inventory ...")
    inv = inventory(ex)
    ids = [i for i in inv if inv[i][0] and inv[i][1]]  # ids with BOTH classes
    rng.shuffle(ids)
    log(f"[OK] {len(ids)} identities with both live+spoof (of {len(inv)} total)")

    n_T = max(2, int(len(ids) * args.test_frac_ids))
    T_ids = set(ids[:n_T])
    non_T = ids[n_T:]

    cap = args.cap_per_id

    # TEST set: from each T id take up to test_per_id (balanced live/spoof)
    test = []
    for i in T_ids:
        for lab in (0, 1):
            imgs = inv[i][lab][:]
            rng.shuffle(imgs)
            for p in imgs[:max(1, args.test_per_id // 2)]:
                test.append({"path": p, "label": lab, "identity": i})
    test_paths = {t["path"] for t in test}

    # L0-train: non-T images (cap per id), balanced to subsample
    def pool_from(id_list, exclude=None):
        by = {0: [], 1: []}
        for i in id_list:
            for lab in (0, 1):
                for p in inv[i][lab][:cap]:
                    if exclude and p in exclude:
                        continue
                    by[lab].append((p, lab, i))
        return by

    l0_by = pool_from(non_T)
    per = args.subsample // 2
    for lab in (0, 1):
        rng.shuffle(l0_by[lab])
    n_each = min(per, len(l0_by[0]), len(l0_by[1]))
    l0 = [{"path": p, "label": l} for (p, l, _i) in (l0_by[0][:n_each] + l0_by[1][:n_each])]
    l0_ids = [_i for (_p, _l, _i) in (l0_by[0][:n_each] + l0_by[1][:n_each])]

    # NAIVE-train: pool from ALL ids (incl T's non-test imgs) excluding test, same size
    naive_by = pool_from(ids, exclude=test_paths)
    for lab in (0, 1):
        rng.shuffle(naive_by[lab])
    naive = [{"path": p, "label": l} for (p, l, _i) in (naive_by[0][:n_each] + naive_by[1][:n_each])]
    naive_ids = [_i for (_p, _l, _i) in (naive_by[0][:n_each] + naive_by[1][:n_each])]

    rng.shuffle(naive); rng.shuffle(l0)
    test_ids = [t["identity"] for t in test]
    Ln = leakage_metrics(naive_ids, test_ids)
    Ll = leakage_metrics(l0_ids, test_ids)
    log(f"[OK] splits: naive={len(naive)} l0={len(l0)} test={len(test)} "
        f"| L(naive)={Ln['L_id_over_test']:.3f} L(l0)={Ll['L_id_over_test']:.3f}")
    return naive, l0, test, Ln, Ll


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def paired_ci(y, pn, pl, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y); ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        ds.append(auc(y[idx], pn[idx]) - auc(y[idx], pl[idx]))
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=40000, help="images per train arm")
    ap.add_argument("--cap-per-id", type=int, default=40)
    ap.add_argument("--test-frac-ids", type=float, default=0.20)
    ap.add_argument("--test-per-id", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shutdown", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.subsample, a.cap_per_id, a.test_frac_ids, a.test_per_id, a.epochs = 800, 6, 0.25, 4, 1
        a.shutdown = False

    os.makedirs(OUTD, exist_ok=True)
    open(LOG, "w").close()
    t0 = time.time()
    ntfy("Effort leakage A/B START", f"smoke={a.smoke} epochs={a.epochs} subsample={a.subsample}")
    try:
        status("build_splits")
        naive, l0, test, Ln, Ll = build_splits(a)
        f_naive = os.path.join(OUTD, "leak_train_naive.json")
        f_l0 = os.path.join(OUTD, "leak_train_l0.json")
        f_testp = os.path.join(OUTD, "leak_test_paths.json")
        json.dump(naive, open(f_naive, "w"))
        json.dump(l0, open(f_l0, "w"))
        json.dump([t["path"] for t in test], open(f_testp, "w"))

        # preflight on the naive list (fail fast before the long run)
        status("preflight")
        run([PY, "_dodo_effort_train.py", "--train-list", f_naive, "--out",
             os.path.join(OUTD, "_pf.pth"), "--batch", str(a.batch),
             "--device", a.device, "--preflight"], "preflight")

        ck_naive = os.path.join(OUTD, "ckpt_naive.pth")
        ck_l0 = os.path.join(OUTD, "ckpt_l0.pth")
        common = ["--epochs", str(a.epochs), "--lr", str(a.lr), "--batch", str(a.batch),
                  "--workers", str(a.workers), "--device", a.device, "--seed", str(a.seed)]
        status("train_naive")
        run([PY, "_dodo_effort_train.py", "--train-list", f_naive, "--out", ck_naive] + common, "train_naive")
        status("train_l0")
        run([PY, "_dodo_effort_train.py", "--train-list", f_l0, "--out", ck_l0] + common, "train_l0")

        sc_naive = os.path.join(OUTD, "scores_naive.json")
        sc_l0 = os.path.join(OUTD, "scores_l0.json")
        status("score_naive")
        run([PY, "_dodo_effort_score.py", "--weights", ck_naive, "--inputs", f_testp,
             "--out", sc_naive, "--device", a.device], "score_naive")
        status("score_l0")
        run([PY, "_dodo_effort_score.py", "--weights", ck_l0, "--inputs", f_testp,
             "--out", sc_l0, "--device", a.device], "score_l0")

        s_n = json.load(open(sc_naive)); s_l = json.load(open(sc_l0))
        y, pn, pl = [], [], []
        for t in test:
            if t["path"] in s_n and t["path"] in s_l:
                y.append(t["label"]); pn.append(s_n[t["path"]]); pl.append(s_l[t["path"]])
        y = np.array(y); pn = np.array(pn); pl = np.array(pl)
        auc_n, auc_l = auc(y, pn), auc(y, pl)
        d = auc_n - auc_l
        lo, hi = paired_ci(y, pn, pl)
        res = {
            "delta_auc_naive_minus_l0": d, "ci95": [lo, hi],
            "auc_naive": auc_n, "auc_l0": auc_l,
            "L_naive": Ln["L_id_over_test"], "L_l0": Ll["L_id_over_test"],
            "n_test": int(len(y)), "n_train_each": len(naive),
            "epochs": a.epochs, "smoke": a.smoke,
            "minutes": round((time.time() - t0) / 60, 1),
            "verdict": ("LEAKAGE (CI excludes 0, positive)" if lo > 0 else
                        "no significant leakage (CI includes 0)"),
        }
        json.dump(res, open(RESULT, "w"), indent=2)
        status("done", **res)
        log("=" * 55)
        log(f"[RESULT] delta-AUC = {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")
        log(f"[RESULT] AUC naive={auc_n:.4f}  AUC L0={auc_l:.4f}  | L {Ln['L_id_over_test']:.2f} vs {Ll['L_id_over_test']:.2f}")
        log(f"[VERDICT] {res['verdict']}  ({res['minutes']} min)")
        ntfy("Effort leakage A/B DONE",
             f"delta-AUC {d:+.4f} CI[{lo:+.3f},{hi:+.3f}] | naive {auc_n:.3f} vs L0 {auc_l:.3f} | {res['verdict']}")
    except Exception as e:
        log(f"[ERR] run failed: {e}")
        log(traceback.format_exc())
        status("FAILED", error=str(e))
        ntfy("Effort leakage A/B FAILED", str(e)[:180])
    finally:
        if a.shutdown:
            log("[..] shutdown /s /t 90 (cancel: shutdown /a)")
            os.system('shutdown /s /t 90 /c "DodoVisionAI: leakage A/B ended. Shutdown in 90s; shutdown /a to cancel."')


if __name__ == "__main__":
    main()
