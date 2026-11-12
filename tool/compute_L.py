"""Compute asymmetric train/test identity overlap from explicit JSON inputs.

Use --key celeba, client or video to choose the path naming convention.
No dataset records or study results are bundled with this code-only snapshot.
The metric divides unique shared identities by max(1, unique test identities).
"""
import argparse
import json
import os
import re
import sys


def load_paths(fname):
    data = json.load(open(fname, encoding="utf8"))
    out = []
    for item in data:
        out.append(item["path"] if isinstance(item, dict) else str(item))
    return out


def key_celeba(path):
    """CelebA-Spoof crop rule: the digit prefix of the file name
    (10000_019598.jpg -> 10000)."""
    stem = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0]
    m = re.match(r"^(\d+)_\d+$", stem)
    return m.group(1) if m else stem.split("_")[0]


def key_client(path):
    """Replay-Attack rule: the clientNNN token anywhere in the path."""
    m = re.search(r"client(\d+)", path)
    if not m:
        raise SystemExit(f"[ERR] no client token in {path!r}")
    return "client" + m.group(1)


def key_video(path):
    """Video rule: the file stem minus a trailing frame index
    (_fNNNNN or __fNNNNN)."""
    stem = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0]
    return re.sub(r"_+f\d+$", "", stem)


KEYS = {"celeba": key_celeba, "client": key_client, "video": key_video}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--key", choices=sorted(KEYS), required=True)
    a = ap.parse_args()
    fn = KEYS[a.key]
    train_ids = set(fn(p) for p in load_paths(a.train))
    test_ids = set(fn(p) for p in load_paths(a.test))
    inter = train_ids & test_ids
    L = len(inter) / max(1, len(test_ids))
    print(f"[L] key={a.key}  train_ids={len(train_ids)}  "
          f"test_ids={len(test_ids)}  overlap={len(inter)}  L = {L:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
