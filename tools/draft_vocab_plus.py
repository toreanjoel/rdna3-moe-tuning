#!/usr/bin/env python3
# draft_vocab_plus.py — a wider draft vocabulary for the MTP head: a base subset (the fork's zh60k) plus the tokens this model really writes (token-id lists from its own answers / coding sessions) and/or every id below a bound; prints coverage of held-out token counts
# usage (engine venv): draft_vocab_plus.py --base engine/rdna3/draft_vocab_zh60k.bin [--add ids.npy ...] [--below 100000] --out data/mtp/draft_vocab_plus.bin
#        ids.npy: an int array of token ids, or (id, count) rows (first column used).  The base ids keep their order; the added ones follow in id order.
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--add", nargs="*", default=[]); ap.add_argument("--below", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = np.fromfile(a.base, dtype=np.int32)
have = set(base.tolist()); extra = set(range(a.below))
for f in a.add:
    x = np.load(f); extra |= set((x[:, 0] if x.ndim == 2 else x).astype(np.int64).tolist())
new = np.array(sorted(extra - have), dtype=np.int32)
np.concatenate([base, new]).tofile(a.out)
print(f"{len(base)} base ids + {len(new)} added = {len(base) + len(new)} -> {a.out}")
