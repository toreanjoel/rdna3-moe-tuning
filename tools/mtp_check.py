#!/usr/bin/env python3
# mtp_check.py — train/serve mismatch check for the MTP draft layer: the torch reference (tools/mtp_train.py) on the rows the server really drafts from (STRATA_MTP_DUMP_DEC) vs the prompt-path rows of the same text, vs the server's own drafts, with and without the run-time quantization (Q4_0 projections / experts / head)
# usage (train-venv, server stopped): mtp_check.py --pre PRE.bin --dec DEC.bin [--replay REPLAY.bin] --ckpt C --vocab zh60k.bin [--rt-vocab rt/draft_vocab.bin] [--q4]
#   PRE = STRATA_MTP_DUMP of the run that wrote DEC (prompt rows of each request); REPLAY = STRATA_MTP_DUMP of tools/mtp_data.py replay over the same answers
import argparse, struct, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_train as M
from mtp_train import H, HC, DEV

from mtp_train import q4_0, q4_big

def read_dec(path):
    """-> requests: list of rounds (p, T, a, x, out tokens, drafts, probs, offset of the (a + 1) bf16 rows)."""
    reqs, cur, last = [], None, None
    with open(path, "rb") as f:
        size = f.seek(0, 2); off = 0
        while off + 36 <= size:
            f.seek(off); hdr = f.read(36)
            assert hdr[:4] == b"MTPV", f"bad record at {off}"
            p, T, a, n = struct.unpack("<qqqq", hdr[4:])
            x = struct.unpack("<i", f.read(4))[0]
            out = np.frombuffer(f.read(4 * (a + 1)), dtype=np.int32).astype(np.int64)
            dr = np.frombuffer(f.read(4 * n), dtype=np.int32).astype(np.int64)
            pr = np.frombuffer(f.read(4 * n), dtype=np.float32)
            roff = off + 36 + 4 + 4 * (a + 1) + 8 * n
            nb = (a + 1) * HC * H * 2
            if roff + nb > size: break
            if last is None or p != last:   # not the continuation of the request: a new one
                cur = []; reqs.append(cur)
            cur.append((p, T, a, x, out, dr, pr, roff))
            last = p + a + 1
            off = roff + nb
    return reqs

def rows_at(mm, off, n):
    u = np.asarray(mm[off // 2: off // 2 + n * HC * H]).reshape(n, HC * H)
    return torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))

@torch.no_grad()
def steps_acc(m, R, toks, mask, lo, steps=3):
    """per-step top-1 on the cells >= lo (accepted path, as mtp_train.evaluate) + the step-1 predictions and probabilities."""
    toks_t = torch.from_numpy(toks)
    n = min(R.shape[0], len(toks) - 2)
    pos = torch.arange(n, device=DEV)
    sample, Rp, (k, v) = m(R[:n].to(DEV), toks_t[1:n + 1].to(DEV), pos)
    lg = (sample.to(torch.bfloat16) @ m.head.T).float() + (mask if mask is not None else 0)
    sm = torch.softmax(lg, -1)
    prob1, pred1 = sm.max(-1)
    lab = toks_t[2:n + 2].to(DEV)
    hit = [((pred1 == lab)[lo:]).sum().item()]; tot = [max(n - lo, 0)]
    for j in range(1, steps):
        if n - j <= lo: break
        tn_j = toks_t[j + 1: j + 1 + n - j].to(DEV); lab_j = toks_t[j + 2: j + 2 + n - j].to(DEV)
        L = len(lab_j)
        sample, Rp, _ = m(Rp[: n - j][:L], tn_j[:L], pos[: n - j][:L] + j, kv=(k, v, pos))
        pj = ((sample.to(torch.bfloat16) @ m.head.T).float() + (mask if mask is not None else 0)).argmax(-1)
        hit.append(((pj == lab_j)[lo:]).sum().item()); tot.append(max(L - lo, 0))
    return hit, tot, pred1.cpu().numpy(), prob1.cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True); ap.add_argument("--dec", required=True); ap.add_argument("--replay")
    ap.add_argument("--ckpt"); ap.add_argument("--vocab"); ap.add_argument("--rt-vocab"); ap.add_argument("--w", default=str(M.D / "train"))
    ap.add_argument("--rt", default=str(M.D / "rt-q4_0")); ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--q4", action="store_true", help="also evaluate with the run-time Q4_0 quantization applied")
    a = ap.parse_args()
    pre = M.read_dump_one(a.pre); dec = read_dec(a.dec)
    rep = M.read_dump_one(a.replay) if a.replay else []
    mm_pre = np.memmap(a.pre, dtype=np.uint16, mode="r"); mm_dec = np.memmap(a.dec, dtype=np.uint16, mode="r")
    print(f"{len(pre)} prompt sequences, {len(dec)} decode requests, {len(rep)} replay sequences")
    # pair each decode request (first cell p = n - 1) with the next prompt sequence of n tokens
    cases, i = [], 0
    for rq in dec:
        n = rq[0][0] + 1
        j = next((k for k in range(i, min(i + 3, len(pre))) if len(pre[k][0]) == n), None)   # (a very short prompt leaves no prompt record)
        if j is None: continue
        toks, chunks = pre[j]; i = j + 1
        if sum(c[1] for c in chunks) < n - 1 or sum(r[2] + 1 for r in rq) < 32: continue
        Rpre = torch.cat([rows_at(mm_pre, o, T) for _, T, o in chunks])[: n - 1]
        Rdec = torch.cat([rows_at(mm_dec, r[7], r[2] + 1) for r in rq])
        gen = np.concatenate([r[4] for r in rq])
        assert rq[0][3] == toks[n - 1], "the first window's token is not the prompt's last"
        cases.append((n, torch.cat([Rpre, Rdec]), np.concatenate([toks, gen]), rq))
    print(f"{len(cases)} requests paired, {sum(len(c[2]) - c[0] for c in cases)} generated tokens")
    m = M.make_model(a); m.eval()
    V = m.head.shape[0]
    def mask_of(path):
        return M.vocab_mask(argparse.Namespace(vocab=path), V) if path else None
    mask = mask_of(a.vocab)
    if a.rt_vocab:
        ids = lambda f: set(np.fromfile(f, dtype=np.int32).tolist())
        sv, rv = ids(a.vocab), ids(a.rt_vocab)
        d0 = [int(r[5][0]) for c in cases for r in c[3] if len(r[5])]
        print(f"draft vocab: --vocab {len(sv)} ids, rt dir {len(rv)} ids, shared {len(sv & rv)}; server step-1 drafts {len(d0)}: "
              f"{sum(t in sv for t in d0)} in --vocab, {sum(t in rv for t in d0)} in the rt dir's")
        lab = [int(t) for c in cases for t in c[2][c[0] + 1:]]
        print(f"generated tokens the draft vocab cannot emit: --vocab {sum(t not in sv for t in lab) / len(lab):.2%}, rt dir {sum(t not in rv for t in lab) / len(lab):.2%}")
    def run(tag):
        H3, T3 = [0, 0, 0], [0, 0, 0]; agree = same_tok = nd = 0; dp = []; R3, RT3 = [0, 0, 0], [0, 0, 0]; cos = []; tokeq = [0, 0]
        ri = 0
        for n, R, toks, rq in cases:
            hit, tot, pred1, prob1 = steps_acc(m, R, toks, mask, n - 1)
            for j in range(len(hit)): H3[j] += hit[j]; T3[j] += tot[j]
            for p, T, a_, x, out, dr, pr, _ in rq:   # the server's chain from cell p + a: its first draft is the token at p + a + 2
                c = p + a_
                if len(dr) == 0 or c >= len(pred1): continue
                nd += 1; agree += int(pred1[c] == dr[0]); dp.append(abs(float(prob1[c]) - float(pr[0])))
            # the same text through the prompt path (replay): accuracy on the answer cells, and how far the rows are apart
            while ri < len(rep) and not (len(rep[ri][0]) >= n and np.array_equal(rep[ri][0][:n - 5], toks[:n - 5])): ri += 1
            if ri < len(rep):
                rt_, rc = rep[ri]; ri += 1
                Rr = torch.cat([rows_at(np.memmap(a.replay, dtype=np.uint16, mode="r"), o, T) for _, T, o in rc])
                L = min(len(rt_), len(toks)); eq = int((rt_[:L] == toks[:L]).sum()); tokeq[0] += eq; tokeq[1] += L
                h2, t2, _, _ = steps_acc(m, Rr, rt_, mask, n - 1)
                for j in range(len(h2)): R3[j] += h2[j]; RT3[j] += t2[j]
                if eq == L:
                    k = min(Rr.shape[0], R.shape[0])
                    cs = torch.nn.functional.cosine_similarity(Rr[n - 1:k].float(), R[n - 1:k].float(), dim=-1)
                    rel = ((Rr[n - 1:k] - R[n - 1:k]).float().norm(dim=-1) / R[n - 1:k].float().norm(dim=-1))
                    cos.append((cs.mean().item(), rel.mean().item()))
        f = lambda h, t: " / ".join(f"{x / max(y, 1):.3f}" for x, y in zip(h, t))
        print(f"[{tag}] torch head on the DECODE rows, steps 1-3 top-1: {f(H3, T3)}  ({T3[0]} cells)")
        if RT3[0]: print(f"[{tag}] torch head on the PROMPT-path rows of the same text:  {f(R3, RT3)}  ({RT3[0]} cells; replay tokens equal {tokeq[0] / max(tokeq[1], 1):.3f})")
        if cos: print(f"[{tag}] decode vs prompt-path rows: cosine {np.mean([c[0] for c in cos]):.5f}, relative difference {np.mean([c[1] for c in cos]):.4f}")
        print(f"[{tag}] torch step-1 pick = the server's first draft: {agree / max(nd, 1):.3f} of {nd} chains; mean |p torch - p server| {np.mean(dp) if dp else 0:.3f}")
    run("float")
    if a.q4:
        kinds = M.rt_kinds(a.rt)
        with torch.no_grad():
            for k, p in m.p.items():
                if kinds.get(k.replace("__", ".")) == "q8_0": p.copy_(q4_0(p))
            run("q4 projections")
            q4_big(m.gu); q4_big(m.dn)
            run("q4 projections + experts")
            q4_big(m.head, 4096)
            run("q4 projections + experts + head")

if __name__ == "__main__":
    main()
