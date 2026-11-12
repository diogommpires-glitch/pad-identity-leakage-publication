"""pad/leakage_metric.py -- continuous train/test leakage metric L in [0,1].

v2026-06-21 (benchmark-integrity scaffold, piece 1/5).

The pre-existing `pad/split_leakage_audit.py` is a BINARY gate: a split
PASSes iff `intersection_count == 0`. That is the right CI gate, but the
killer experiment needs a CONTINUOUS leakage coordinate L in [0,1] so a
detector can be scored along the axis (naive split -> high L; group-disjoint
split -> L = 0) and the AUC drop plotted against it.

This module ADDS the continuous metric WITHOUT touching the binary gate:
  - `leakage_metrics(train_ids, val_ids)` -> the pure, manifest-free metric
    dict (L_id over test, L_id over train, symmetric Jaccard, intersection
    count, sizes). Computable for ANY (train, test) id pair.
  - identity-extractor REGISTRY so any dataset/record style yields a group id
    via one call -- reuses `pad.metrics.group_id_for_record` (the same key a
    reviewer would check) and the patch_depth filename rule.
  - `split_leakage_L(train_records, test_records, ...)` -> L for two record
    lists (the form the killer runner consumes).
  - `permutation_null_L(...)` -> a label-free significance test: how large is
    the observed L vs the L you would see under random group assignment.
  - `embedding_leakage_L(...)` -> OPTIONAL L_emb(tau): the fraction of test
    items whose nearest train neighbour in a supplied embedding space is
    within distance tau (near-duplicate leakage that survives id-disjointness).
    Pure-numpy, no model load; returns a clear [--] skip dict if no embeddings
    are supplied so it never fails loud on the common (embedding-free) path.

NON-BREAKING: `pad/split_leakage_audit.py` is wrapped, not edited. Its
`stamp_split_manifest` already emits `intersection_count` /
`intersection_fraction` (an asymmetric L over the val set) and its
`audit_all_manifests` verdict still thresholds on `intersection_count == 0`.
Callers and `pad/tests/test_no_split_leakage.py` are unaffected.

CLI:
    python -m pad.leakage_metric --train recsA.jsonl --test recsB.jsonl
    python -m pad.leakage_metric --records one.jsonl --naive   # naive vs L0 demo
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Reuse the canonical group/identity derivation -- the SAME key the split
# auditor and the bootstrap CI grouping use, so L lines up with the manifest.
from pad.metrics import group_id_for_record

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── identity-extractor registry ───────────────────────────────────────
#
# An extractor maps ONE record (dict) -> a group/identity string. Every
# dataset style the repo handles routes through group_id_for_record, but we
# expose a named registry so a future detector with a bespoke id scheme can
# register its own without editing this file.

def _extract_generic(record: Mapping) -> str:
    """Default: the repo-canonical group id (path -> subject/video/utterance)."""
    return group_id_for_record(record)


def _extract_celebdf_subject(record: Mapping) -> str:
    """Celeb-DF-v2 subject/synthesis-pair id (id31_0004.mp4 -> id31;
    id29_id34_0002.mp4 -> id29_id34). This is group_id_for_record's path rule."""
    return group_id_for_record(record)


def _extract_celebdf_video(record: Mapping) -> str:
    """Celeb-DF-v2 SOURCE-VIDEO id (keeps the full stem minus a trailing
    zero-padded clip index). Differs from the subject extractor only for
    Celeb-synthesis pairs, where subject-level grouping is the stricter
    (more leakage-conservative) choice. Provided so the scaffold can pick
    the grouping granularity explicitly (gaps item 6)."""
    p = record.get("path") or record.get("key") or record.get("id") or ""
    import os
    base = os.path.splitext(os.path.basename(str(p).replace("\\", "/").rstrip("/")))[0]
    # strip a single trailing zero-padded index (_0004 / _000) -> video stem
    import re
    m = re.match(r"^(.*)_0\d*$", base)
    return m.group(1) if (m and m.group(1)) else base


def _extract_celeba_identity(record: Mapping) -> str:
    """CelebA-Spoof identity from filename stem (10000_019598.jpg -> 10000).

    Mirrors pad.train_patch_depth._identity_from_filename without importing
    torch-heavy train modules at metric time. Falls back to the generic key
    when the path is not in the CelebA-Spoof <id>_<n> shape.
    """
    p = record.get("path") or record.get("key") or record.get("id") or ""
    import os
    import re
    stem = os.path.splitext(os.path.basename(str(p).replace("\\", "/").rstrip("/")))[0]
    m = re.match(r"^(\d+)_\d+$", stem)
    if m:
        return m.group(1)
    return stem.split("_", 1)[0] if "_" in stem else stem


IDENTITY_EXTRACTORS: Dict[str, Callable[[Mapping], str]] = {
    "generic": _extract_generic,
    "celebdf": _extract_celebdf_subject,
    "celebdf_subject": _extract_celebdf_subject,
    "celebdf_video": _extract_celebdf_video,
    "celeba": _extract_celeba_identity,
    "ffpp": _extract_generic,
    "replay": _extract_generic,
    "asvspoof": _extract_generic,
}


def get_extractor(name: Optional[str]) -> Callable[[Mapping], str]:
    """Return the identity extractor for a dataset name (default: generic).

    FAIL LOUD on an unknown explicit name so a typo never silently falls
    back to a different grouping (which would silently change L).
    """
    if name is None:
        return IDENTITY_EXTRACTORS["generic"]
    key = name.lower()
    if key not in IDENTITY_EXTRACTORS:
        raise KeyError(
            f"[ERR] unknown identity extractor {name!r}; "
            f"known = {sorted(IDENTITY_EXTRACTORS)}")
    return IDENTITY_EXTRACTORS[key]


# ─── the continuous metric (pure, manifest-free) ───────────────────────


def leakage_metrics(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> Dict[str, float]:
    """Continuous leakage metrics for an arbitrary (train, test) id pair.

    All in [0,1]; computable WITHOUT writing a manifest, so the scaffold can
    score L for any candidate split.

    Returns
    -------
    {
      "L_id_over_test":  |train ∩ test| / |test ids|   (== intersection_fraction
                         in split_leakage_audit; the headline asymmetric L:
                         "what fraction of test identities the model already saw"),
      "L_id_over_train": |train ∩ test| / |train ids|,
      "jaccard":         |train ∩ test| / |train ∪ test|   (symmetric L),
      "intersection_count": int,
      "n_train": int (distinct train ids),
      "n_test":  int (distinct test ids),
    }
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    inter = train_set & val_set
    union = train_set | val_set
    return {
        "L_id_over_test":  len(inter) / max(1, len(val_set)),
        "L_id_over_train": len(inter) / max(1, len(train_set)),
        "jaccard":         len(inter) / max(1, len(union)),
        "intersection_count": len(inter),
        "n_train": len(train_set),
        "n_test":  len(val_set),
    }


def L(train_ids: Sequence[str], val_ids: Sequence[str]) -> float:
    """Scalar leakage coordinate L in [0,1] -- the headline asymmetric metric
    (fraction of TEST identities also present in TRAIN). Convenience wrapper."""
    return leakage_metrics(train_ids, val_ids)["L_id_over_test"]


def split_leakage_L(
    train_records: Sequence[Mapping],
    test_records: Sequence[Mapping],
    extractor: str | Callable[[Mapping], str] = "generic",
) -> Dict[str, float]:
    """L for two RECORD lists (the killer-runner form).

    `extractor` is a registry name (see IDENTITY_EXTRACTORS) or a callable
    record->str. Returns the leakage_metrics dict.
    """
    fn = extractor if callable(extractor) else get_extractor(extractor)
    train_ids = [fn(r) for r in train_records]
    test_ids = [fn(r) for r in test_records]
    out = leakage_metrics(train_ids, test_ids)
    out["extractor"] = getattr(fn, "__name__", str(extractor))
    return out


# ─── permutation null / significance of L ──────────────────────────────


def permutation_null_L(
    train_records: Sequence[Mapping],
    test_records: Sequence[Mapping],
    extractor: str | Callable[[Mapping], str] = "generic",
    n_perm: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Significance test for an observed L.

    Null: train/test membership is independent of identity. We pool all
    records, then repeatedly RE-PARTITION GROUPS at random into a train/test
    split of the same group sizes and recompute L_id_over_test. The p-value is
    P[L_null >= L_observed]. A naive (clip-random) split sits near the null
    mean (high L, not significant -- it is *expected* to leak); a deliberately
    group-disjoint split sits far in the low tail (L = 0, p ~ 1.0).

    Deterministic (seed 42 default).
    """
    fn = extractor if callable(extractor) else get_extractor(extractor)
    train_ids = [fn(r) for r in train_records]
    test_ids = [fn(r) for r in test_records]
    obs = leakage_metrics(train_ids, test_ids)["L_id_over_test"]

    # Pool the DISTINCT groups; preserve the observed #train / #test split.
    all_groups = sorted(set(train_ids) | set(test_ids))
    n_total = len(all_groups)
    n_test_groups = len(set(test_ids))
    if n_total < 2 or n_test_groups == 0 or n_test_groups >= n_total:
        return {"L_observed": obs, "p_value": float("nan"),
                "null_mean": float("nan"), "null_std": float("nan"),
                "n_perm": 0, "note": "[--] degenerate group pool; null undefined"}

    rng = random.Random(seed)
    null = np.empty(n_perm, dtype=np.float64)
    # A pure group repartition always has intersection 0 (degenerate null), so
    # the informative null is at the RECORD level: shuffle which records (hence
    # which group-instances) land in test, preserving the observed test size.
    # An overlapping-id (leaky) split sits near the null mean (expected to
    # leak); a group-disjoint split sits far in the low tail (L=0, p~1.0).
    pooled = list(train_records) + list(test_records)
    n_test_recs = len(test_records)
    idx = list(range(len(pooled)))
    for i in range(n_perm):
        rng.shuffle(idx)
        t_idx = idx[:n_test_recs]
        tr_idx = idx[n_test_recs:]
        t_ids = [fn(pooled[k]) for k in t_idx]
        tr_ids = [fn(pooled[k]) for k in tr_idx]
        null[i] = leakage_metrics(tr_ids, t_ids)["L_id_over_test"]

    p = float((null >= obs).mean())
    return {
        "L_observed": float(obs),
        "p_value": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perm": int(n_perm),
        "note": ("[OK] L below random-shuffle null (group-disjoint)" if p > 0.5
                 else "[OK] L at/above null (leaky split, as expected)"),
    }


# ─── optional label-free embedding leakage L_emb(tau) ──────────────────


def embedding_leakage_L(
    train_emb: Optional[np.ndarray],
    test_emb: Optional[np.ndarray],
    tau: float = 0.1,
    metric: str = "cosine",
) -> Dict[str, float]:
    """L_emb(tau): fraction of TEST items whose nearest TRAIN neighbour is
    within distance tau in a supplied embedding space.

    This catches NEAR-DUPLICATE leakage that identity-disjointness misses
    (e.g. two different ids that are visually near-identical frames). It is
    OPTIONAL: when no embeddings are supplied this returns a clear skip dict
    rather than failing -- the id-based L is the primary metric.

    Pure numpy (chunked), no model load. `metric` in {"cosine","euclidean"}.
    For cosine, distance = 1 - cos_sim (rows L2-normalised internally).
    """
    if train_emb is None or test_emb is None:
        return {"L_emb": float("nan"), "tau": tau, "metric": metric,
                "n_test": 0, "note": "[--] no embeddings supplied; L_emb skipped"}
    tr = np.asarray(train_emb, dtype=np.float64)
    te = np.asarray(test_emb, dtype=np.float64)
    if tr.ndim != 2 or te.ndim != 2 or tr.shape[1] != te.shape[1]:
        raise ValueError(
            f"[ERR] embedding shape mismatch train={tr.shape} test={te.shape}")
    if metric == "cosine":
        tr = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-12)
        te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-12)
    within = 0
    nn_dists = np.empty(len(te), dtype=np.float64)
    CH = 4096
    for s in range(0, len(te), CH):
        blk = te[s:s + CH]
        if metric == "cosine":
            d = 1.0 - blk @ tr.T               # (b, n_train)
        elif metric == "euclidean":
            # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
            d = (np.sum(blk ** 2, axis=1, keepdims=True)
                 + np.sum(tr ** 2, axis=1)[None, :]
                 - 2.0 * blk @ tr.T)
            d = np.sqrt(np.maximum(d, 0.0))
        else:
            raise ValueError(f"[ERR] unknown metric {metric!r}")
        mind = d.min(axis=1)
        nn_dists[s:s + len(blk)] = mind
        within += int((mind <= tau).sum())
    return {
        "L_emb": within / max(1, len(te)),
        "tau": float(tau),
        "metric": metric,
        "n_test": int(len(te)),
        "nn_dist_median": float(np.median(nn_dists)),
        "nn_dist_p05": float(np.percentile(nn_dists, 5)),
        "note": "[OK] embedding near-duplicate leakage computed",
    }


# ─── naive vs L=0 partition builders (single-file killer experiment) ────


def naive_index_split(
    records: Sequence[Mapping],
    val_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """LEAKY split: global index shuffle then cut (no grouping). Reproduces
    pad.diagnose_split_leakage.old_index_split semantics. Groups straddle the
    train/test boundary -> high L."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(len(records) * val_fraction))
    return idx[cut:], idx[:cut]   # (train, val)


def group_disjoint_split(
    records: Sequence[Mapping],
    val_fraction: float = 0.5,
    seed: int = 42,
    extractor: str | Callable[[Mapping], str] = "generic",
) -> Tuple[List[int], List[int]]:
    """L=0 split: shuffle GROUPS, cut val_fraction of GROUPS into val. No group
    appears on both sides -> L = 0 by construction. Mirrors
    train_patch_depth._identity_stratified_split (group-level cut)."""
    fn = extractor if callable(extractor) else get_extractor(extractor)
    by_g: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_g.setdefault(fn(r), []).append(i)
    groups = sorted(by_g)
    random.Random(seed).shuffle(groups)
    cut = max(1, int(len(groups) * val_fraction))
    val_g = set(groups[:cut])
    train_idx, val_idx = [], []
    for g, members in by_g.items():
        (val_idx if g in val_g else train_idx).extend(members)
    return sorted(train_idx), sorted(val_idx)


# ─── CLI ────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"[ERR] missing records file: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, help="train-side records jsonl")
    ap.add_argument("--test", type=Path, help="test-side records jsonl")
    ap.add_argument("--records", type=Path,
                    help="single records jsonl for the naive-vs-L0 demo")
    ap.add_argument("--extractor", default="generic",
                    help=f"identity extractor: {sorted(IDENTITY_EXTRACTORS)}")
    ap.add_argument("--naive", action="store_true",
                    help="with --records: print naive(leaky) vs group-disjoint L")
    ap.add_argument("--val-fraction", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.records and args.naive:
        recs = _load_jsonl(args.records)
        tr_n, va_n = naive_index_split(recs, args.val_fraction, args.seed)
        tr_g, va_g = group_disjoint_split(recs, args.val_fraction, args.seed,
                                          args.extractor)
        m_naive = split_leakage_L([recs[i] for i in tr_n], [recs[i] for i in va_n],
                                  args.extractor)
        m_clean = split_leakage_L([recs[i] for i in tr_g], [recs[i] for i in va_g],
                                  args.extractor)
        print(f"[leakage] file={args.records.name}  n={len(recs)}  "
              f"extractor={args.extractor}")
        print(f"[leakage] NAIVE  (index split): L_id={m_naive['L_id_over_test']:.4f} "
              f"jaccard={m_naive['jaccard']:.4f} inter={m_naive['intersection_count']}")
        print(f"[leakage] L=0    (group split): L_id={m_clean['L_id_over_test']:.4f} "
              f"jaccard={m_clean['jaccard']:.4f} inter={m_clean['intersection_count']}")
        print(f"[leakage] delta-L = {m_naive['L_id_over_test'] - m_clean['L_id_over_test']:+.4f}")
        return 0

    if args.train and args.test:
        tr = _load_jsonl(args.train)
        te = _load_jsonl(args.test)
        m = split_leakage_L(tr, te, args.extractor)
        perm = permutation_null_L(tr, te, args.extractor, args.n_perm, args.seed)
        print(f"[leakage] train={args.train.name}  test={args.test.name}  "
              f"extractor={args.extractor}")
        print(f"[leakage] L_id_over_test={m['L_id_over_test']:.4f}  "
              f"L_id_over_train={m['L_id_over_train']:.4f}  "
              f"jaccard={m['jaccard']:.4f}")
        print(f"[leakage] n_train_ids={m['n_train']}  n_test_ids={m['n_test']}  "
              f"intersection={m['intersection_count']}")
        print(f"[leakage] permutation null: L_obs={perm['L_observed']:.4f}  "
              f"null_mean={perm['null_mean']:.4f}  p={perm['p_value']:.4f}  "
              f"{perm['note']}")
        return 0

    ap.error("supply either (--train AND --test) or (--records AND --naive)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
