"""Compare filename and client-token identity keys for explicit Replay-style paths.

Supply the JSON path-array as the first argument. The original audit pool is
not distributed, so no original study counts are reproduced by this snapshot.
"""
import json, random, re, sys

paths = json.load(open(sys.argv[1] if len(sys.argv) > 1
                       else "audit_pools/replay_audit_pool.json"))

def zeropad_stem(p):
    stem = p.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.match(r"^(.*)_0\d*$", stem)
    return m.group(1) if m else stem

def client(p):
    return "client" + re.search(r"client(\d+)", p).group(1)

idx = list(range(len(paths)))
random.Random(42).shuffle(idx)
cut = max(1, int(len(paths) * 0.5))
val, train = idx[:cut], idx[cut:]
for name, key in (("generic zero-pad stem", zeropad_stem), ("client", client)):
    tr = set(key(paths[i]) for i in train)
    te = set(key(paths[i]) for i in val)
    print(f"[{name}] distinct ids = {len(set(key(p) for p in paths))}  "
          f"L_naive = {len(tr & te) / len(te):.3f}")

tr_off, te_off = set(), set()
for p in paths:
    base = p.replace("\\", "/").rsplit("/", 1)[-1]
    (tr_off if base.startswith("train_") else te_off).add(client(p))
print(f"[official protocol] train clients = {len(tr_off)}  "
      f"test clients = {len(te_off)}  intersection = {len(tr_off & te_off)}  "
      f"L_official = {len(tr_off & te_off) / len(te_off):.3f}")
