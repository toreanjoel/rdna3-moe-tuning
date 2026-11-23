#!/usr/bin/env python3
# mtp_rt_native.py — MTP runtime dir with the 512 experts in the native layout (gate rows, up rows, down rows per expert) + experts.txt, for patches/mtp-native.patch
# usage (engine venv): python mtp_rt_native.py --gguf data/mtp/mtp-q8_0.gguf --base data/mtp/rt --out data/mtp/rt-q8_0
# The dense weights and draft vocabulary are the base rt dir's (they don't depend on the expert format).
import argparse, shutil
from pathlib import Path
import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
r = gguf.GGUFReader(a.gguf)
t = {x.name: x for x in r.tensors}
gu_t, dn_t = t["mtp.layers.0.mlp.experts.gate_up_proj"], t["mtp.layers.0.mlp.experts.down_proj"]
gu, dn = np.asarray(gu_t.data), np.asarray(dn_t.data)   # (512, 1280, row bytes) gate rows 0..639 then up; (512, 2560, row bytes)
NE, FF2, _ = gu.shape
assert dn.shape[0] == NE and FF2 % 2 == 0, (gu.shape, dn.shape)
with open(out / "experts.bin", "wb") as f:
    for e in range(NE):
        f.write(np.ascontiguousarray(gu[e]).tobytes()); f.write(np.ascontiguousarray(dn[e]).tobytes())
(out / "experts.txt").write_text(f"{int(gu_t.tensor_type)} {int(dn_t.tensor_type)}\n")
for n in ("dense.bin", "dense.txt", "draft_vocab.bin"):
    shutil.copyfile(Path(a.base) / n, out / n)
print(f"experts.bin {(out / 'experts.bin').stat().st_size} B ({NE} x {gu[0].nbytes + dn[0].nbytes}), types {int(gu_t.tensor_type)} {int(dn_t.tensor_type)} -> {out}")
