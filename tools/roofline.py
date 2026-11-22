#!/usr/bin/env python3
# roofline.py — bytes a decode step must read per stage (from a gguf_tensors.py dump) vs the measured GPU ms per stage -> floor ms and headroom at a given DRAM bandwidth
# usage: python3 -I tools/roofline.py tensors.txt [--tokens 2.8] [--ctx 8192] [--bw 960] [--eff 0.85] [--hit 0.9] [--hc-bytes 1]
import argparse, re
from collections import defaultdict

BPW = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5, "IQ4_XS": 4.25,
       "IQ4_NL": 4.5, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "IQ2_S": 2.5625, "IQ2_XS": 2.3125, "Q2_K": 2.625, "Q2_0": 2.5}
# measured GPU ms per step, 8K (snapshot/2026-11-22/sweep6-profile.txt); draft = MTP head + its layer
MEASURED = {"gdn/attention layers": 7.6, "experts (VRAM)": 4.1, "hyper-connections": 3.4, "router + shared expert": 2.8,
            "copy + combine": 2.0, "output head": 0.7, "waits": 1.1, "draft (MTP)": 1.7}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--tokens", type=float, default=2.8, help="verify window tokens")
    ap.add_argument("--ctx", type=int, default=8192); ap.add_argument("--bw", type=float, default=960)
    ap.add_argument("--eff", type=float, default=0.85, help="reachable fraction of --bw")
    ap.add_argument("--hit", type=float, default=0.9, help="expert VRAM hit rate")
    ap.add_argument("--hc-bytes", type=float, default=1, help="bytes/weight of the hyper-connection matrices as run (int8 = 1)")
    a = ap.parse_args()
    st = defaultdict(float); topk, nexp = 10, 512
    for line in open(a.dump):
        m = re.match(r"(\S+)\s+(\S+)\s+\[([\d, ]+)\]", line)
        if not m: continue
        name, typ, shape = m.group(1), m.group(2), [int(x) for x in m.group(3).split(",")]
        n = 1
        for s in shape: n *= s
        b = n * BPW.get(typ, 8) / 8
        if "_exps" in name:
            uniq = nexp * (1 - (1 - topk / nexp) ** a.tokens)          # distinct experts a window touches
            st["experts (VRAM)"] += b / nexp * uniq * a.hit
        elif "hc_" in name and ("down" in name or "up" in name):
            st["hyper-connections"] += n * a.hc_bytes
        elif name.startswith("output_hc"):
            st["hyper-connections"] += n * a.hc_bytes / 48
        elif "gate_inp" in name or "shexp" in name:
            st["router + shared expert"] += b
        elif name == "output.weight":
            st["output head"] += b
        elif name.startswith("blk.") and any(k in name for k in ("attn_", "ssm_", "indexer")):
            st["gdn/attention layers"] += b
    st["gdn/attention layers"] += 12 * a.ctx * (512 + 512 + 128)      # QSA int8 K/V (+ indexer keys), 12 layers
    st["gdn/attention layers"] += 36 * 2 * 32 * 128 * 128 * 4           # GDN fp32 state read + write (est.)
    st["draft (MTP)"] += 60000 * 2560 * 0.5625 + 0.04e9                 # q4 draft head over the 60K vocab + its layer (est.)
    bw = a.bw * a.eff
    print(f"window {a.tokens} tokens, ctx {a.ctx}, {a.bw:.0f} GB/s x {a.eff} = {bw:.0f} GB/s reachable")
    print(f"{'stage':26} {'GB/step':>8} {'floor ms':>9} {'now ms':>7} {'gap ms':>7} {'now GB/s':>9}")
    tf = tn = 0
    for k, now in MEASURED.items():
        gb = st.get(k, 0) / 1e9; fl = gb / bw * 1e3; tf += fl; tn += now
        print(f"{k:26} {gb:8.2f} {fl:9.2f} {now:7.2f} {now - fl:7.2f} {gb / now * 1e3 if gb else 0:9.0f}")
    print(f"{'total':26} {sum(st.values()) / 1e9:8.2f} {tf:9.2f} {tn:7.2f} {tn - tf:7.2f}")
    for keep in (2.1, 2.5, 3.0):
        print(f"tokens kept/step {keep}: now {keep / tn * 1e3:.0f} tok/s; at the floor {keep / tf * 1e3:.0f}; "
              f"floor + 3 ms of other cost {keep / (tf + 3) * 1e3:.0f}")

if __name__ == "__main__":
    main()
