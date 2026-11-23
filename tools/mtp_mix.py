#!/usr/bin/env python3
# mtp_mix.py — MTP head experts at mixed precision, for the "q8_0 on the most-used experts only" test: how often each of the 512 experts is routed, and an rt dir in the q8_0 layout whose cold experts hold the q4_0 pack's values (what a real mixed pack would compute, at q8_0 size)
# usage: mtp_mix.py usage --dump F[,F2] --ckpt C --out usage.npy [--n 200]     (train-venv, server stopped; routing weight per expert)
#        mtp_mix.py pack --usage usage.npy --hot 64 --q4 rt-q4_0 --q8 rt-q8_0 --dense rt-ft --out rt-mix64   (numpy only)
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np

NE, H, FF = 512, 2560, 640
W = 2 * FF * H + H * FF                      # weights per expert: gate + up rows (in H), down rows (in FF)
B4, B8 = W // 32 * 18, W // 32 * 34          # bytes per expert, Q4_0 / Q8_0

def cmd_usage(a):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mtp_train as M
    m = M.make_model(a); m.eval()
    use = torch.zeros(NE, dtype=torch.float64, device=M.DEV); cnt = torch.zeros(NE, dtype=torch.float64, device=M.DEV)
    orig = M.Mtp.experts
    def rec(self, x, ids, wts):
        use.index_add_(0, ids.reshape(-1), wts.reshape(-1).double()); cnt.index_add_(0, ids.reshape(-1), torch.ones(ids.numel(), dtype=torch.float64, device=ids.device))
        return orig(self, x, ids, wts)
    M.Mtp.experts = rec
    seqs = M.read_dump(a.dump)
    with torch.no_grad():
        for s in seqs[-a.n:]:
            R, tn, lab, pos, _ = M.batch(a.dump, s)
            m(R, tn, pos)
    u = (use / use.sum()).cpu().numpy(); c = cnt.cpu().numpy()
    np.save(a.out, u)
    o = np.argsort(-u)
    print(f"{int(c.sum() / 10)} cells; routing weight held by the top 16/32/64/128/256 experts: " +
          " / ".join(f"{u[o[:k]].sum():.3f}" for k in (16, 32, 64, 128, 256)) + f"; experts never routed: {(c == 0).sum()}")

def q4_to_q8(raw):
    """one expert's Q4_0 blocks -> the same values as Q8_0 blocks."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = np.concatenate([b[:, 2:] & 15, b[:, 2:] >> 4], 1).astype(np.float32)
    x = (q - 8.0) * d
    d8 = np.abs(x).max(1) / 127.0
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d8.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = np.round(x / np.where(d8 == 0, 1, d8)[:, None]).clip(-127, 127).astype(np.int8).view(np.uint8)
    return out.tobytes()

def cmd_pack(a):
    u = np.load(a.usage); hot = set(np.argsort(-u)[: a.hot].tolist())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f4 = open(Path(a.q4) / "experts.bin", "rb"); f8 = open(Path(a.q8) / "experts.bin", "rb")
    assert os.fstat(f4.fileno()).st_size == NE * B4 and os.fstat(f8.fileno()).st_size == NE * B8
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            if e in hot: f8.seek(e * B8); f.write(f8.read(B8))
            else: f4.seek(e * B4); f.write(q4_to_q8(f4.read(B4)))
    shutil.copyfile(Path(a.q8) / "experts.txt", out / "experts.txt")
    for n in ("dense.bin", "dense.txt", "draft_vocab.bin"): shutil.copyfile(Path(a.dense) / n, out / n)
    print(f"{a.hot} experts at q8_0 ({u[list(hot)].sum():.1%} of the routing weight), {NE - a.hot} at q4_0 values -> {out}; "
          f"a real mixed pack would be {(a.hot * B8 + (NE - a.hot) * B4) / 2**20:.0f} MiB (q4_0 {NE * B4 / 2**20:.0f}, q8_0 {NE * B8 / 2**20:.0f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--dump"); ap.add_argument("--ckpt"); ap.add_argument("--out"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--w"); ap.add_argument("--rt"); ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--usage"); ap.add_argument("--hot", type=int, default=64); ap.add_argument("--q4"); ap.add_argument("--q8"); ap.add_argument("--dense")
    a = ap.parse_args()
    if a.cmd == "usage":
        import mtp_train as M0
        a.w = a.w or str(M0.D / "train"); a.rt = a.rt or str(M0.D / "rt-q4_0")
    {"usage": cmd_usage, "pack": cmd_pack}[a.cmd](a)
