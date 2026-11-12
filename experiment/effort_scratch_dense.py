"""effort_scratch_dense.py -- the density-controlled inflation A/B.

v1.0 2026-07-12 (all-nighter). The sparse scratch study returned a null
and the diagnosis is in the pool: the published leaky arm carries only
1.79 images per leaked identity, too dilute to memorise. This script
holds L fixed at the published arm's value (the same 1,110 overlapped
test identities) and raises the leak's DENSITY: leaky-dense = those
same identities at ~11 images each (never the test images themselves);
clean-dense = 1,110 fresh non-test identities at the same density.
Same 12,000-image budget, same fixed 15,816-image test, same scratch
CNN, 5 seeds per arm. If inflation appears here, the controlled design
demonstrates it AND identifies the lever: records per leaked identity,
the same quantity Table I's L tracks across benchmarks.

Outputs:
  reports/killer/_pool_dense_{leaky,clean}.json
  reports/killer/_scratchdense_{arm}_s{seed}.json
  reports/killer/scratch_dense_study.json
"""
import io, json, os, random, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from effort_scratch_headroom import KILLER, load_splits, ntfy, run_arm  # noqa: E402

PD = r"E:\pad_datasets\patch_depth"
BUDGET = 12000


def ident(name):
    stem = os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]
    m = re.match(r"^(\d+)_\d+$", stem)
    return m.group(1) if m else stem.split("_")[0]


def index_crops():
    """identity -> {0: [live paths], 1: [spoof paths]} over the full E: set."""
    by_id = defaultdict(lambda: {0: [], 1: []})
    for sub, lab in (("live", 0), ("spoof", 1)):
        d = os.path.join(PD, sub)
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                by_id[ident(f)][lab].append(os.path.join(d, f))
    return by_id


def build_pool(ids, by_id, forbidden, per_id, rng):
    """Sample up to per_id crops per identity, alternating labels for
    balance, skipping forbidden (test) images; trim/fill to BUDGET."""
    recs = []
    for i in ids:
        cand = {lab: [p for p in by_id[i][lab] if p not in forbidden]
                for lab in (0, 1)}
        for lab in (0, 1):
            rng.shuffle(cand[lab])
        take, lab = [], 0
        while len(take) < per_id and (cand[0] or cand[1]):
            if not cand[lab]:
                lab ^= 1
            take.append({"path": cand[lab].pop(), "label": lab})
            lab ^= 1
        recs.extend(take)
    rng.shuffle(recs)
    if len(recs) < BUDGET:
        raise RuntimeError(f"[ERR] pool too small: {len(recs)} < {BUDGET}")
    # trim to budget while keeping label balance as close as possible
    live = [r for r in recs if r["label"] == 0]
    spoof = [r for r in recs if r["label"] == 1]
    half = BUDGET // 2
    n_live = min(len(live), half)
    pool = live[:n_live] + spoof[:BUDGET - n_live]
    rng.shuffle(pool)
    return pool


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--per-id", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=112)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--supplementary-zip")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rng = random.Random(42)

    pools_pub, test_recs = load_splits(a.supplementary_zip)
    test_ids = set(ident(r["path"]) for r in test_recs)
    forbidden = set(r["path"] for r in test_recs)
    overlapped = sorted(set(ident(r["path"]) for r in pools_pub["leaky"]) & test_ids)
    print(f"[OK] {len(overlapped)} overlapped test ids (published leaky arm)", flush=True)

    by_id = index_crops()
    fresh = sorted(i for i in by_id
                   if i not in test_ids
                   and sum(len(v) for v in by_id[i].values()) >= a.per_id)
    rng.shuffle(fresh)
    used_sparse = set(ident(r["path"]) for r in pools_pub["clean"])
    fresh = [i for i in fresh if i not in used_sparse][: len(overlapped)]
    print(f"[OK] {len(fresh)} fresh identities for the clean-dense arm", flush=True)

    leaky_dense = build_pool(overlapped, by_id, forbidden, a.per_id, rng)
    clean_dense = build_pool(fresh, by_id, set(), a.per_id, rng)
    for name, pool in (("leaky", leaky_dense), ("clean", clean_dense)):
        ids = set(ident(r["path"]) for r in pool)
        L = len(ids & test_ids) / len(test_ids)
        live = sum(1 for r in pool if r["label"] == 0)
        print(f"[OK] dense {name}: {len(pool)} imgs, {len(ids)} ids, "
              f"{len(pool)/len(ids):.1f} imgs/id, L = {L:.3f}, "
              f"{live} live / {len(pool)-live} spoof", flush=True)
        with io.open(os.path.join(KILLER, f"_pool_dense_{name}.json"), "w",
                     encoding="utf8") as f:
            json.dump(pool, f)

    ntfy("dense A/B STARTED",
         f"L held at published overlap, ~{len(leaky_dense)/len(overlapped):.0f} "
         f"imgs per leaked id (was 1.79), {len(seeds)} seeds x 2 arms", tags="rocket")

    res = {"leaky": {}, "clean": {}}
    for seed in seeds:
        for arm, pool in (("leaky", leaky_dense), ("clean", clean_dense)):
            auc, _ = run_arm(f"dense_{arm}", pool, test_recs, seed, a)
            res[arm][str(seed)] = auc

    deltas = {s: res["leaky"][str(s)] - res["clean"][str(s)] for s in seeds}
    dv = list(deltas.values())
    placebo = {}
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            placebo[f"{s1}v{s2}"] = res["clean"][str(s1)] - res["clean"][str(s2)]
    leaky_ids = set(ident(r["path"]) for r in leaky_dense)
    summary = {
        "design": "density-controlled A/B: same overlapped identities as the "
                  "published leaky arm (L unchanged), records per leaked "
                  "identity raised from 1.79 to ~11; clean arm matched in "
                  "size, identity count, and density",
        "imgs_per_id_leaky": len(leaky_dense) / len(leaky_ids),
        "L_leaky": len(leaky_ids & test_ids) / len(test_ids),
        "auc_leaky": res["leaky"], "auc_clean": res["clean"],
        "delta_per_seed": {str(k): v for k, v in deltas.items()},
        "delta_mean": float(np.mean(dv)), "delta_sd": float(np.std(dv, ddof=1)),
        "seeds_leaky_higher": int(sum(1 for d in dv if d > 0)),
        "placebo_clean_vs_clean": placebo,
        "placebo_max_abs": float(max(abs(v) for v in placebo.values())),
    }
    out = os.path.join(KILLER, "scratch_dense_study.json")
    with io.open(out, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=1)
    print(f"[OK] DENSE SUMMARY: delta mean {summary['delta_mean']:+.4f} sd "
          f"{summary['delta_sd']:.4f}, leaky higher "
          f"{summary['seeds_leaky_higher']}/{len(seeds)}, placebo max "
          f"{summary['placebo_max_abs']:.4f} -> {out}", flush=True)
    ntfy("dense A/B COMPLETE",
         f"delta mean {summary['delta_mean']:+.4f} sd {summary['delta_sd']:.4f}, "
         f"leaky higher {summary['seeds_leaky_higher']}/{len(seeds)}, "
         f"placebo max {summary['placebo_max_abs']:.4f}", tags="tada")
    return 0


if __name__ == "__main__":
    sys.exit(main())

