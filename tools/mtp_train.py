#!/usr/bin/env python3
# mtp_train.py — distil the MTP draft layer to this IQ3_S backbone: torch reference of the engine's draft layer (src/core/mtp.cpp), trained on STRATA_MTP_DUMP rows, exported as a new rt dir
# usage (train-venv; --qat = with the server's Q4_0 quantization): mtp_train.py prep   --gguf MAIN.gguf --out W          embed + head dequantized once (W/embed.pt, W/head.pt)
#                     mtp_train.py eval   --dump F --w W [--ckpt C]          step-1 top-1 agreement with the backbone's text
#                     mtp_train.py train  --dump F --w W --out C [--steps N --lr X]
#                     mtp_train.py export --ckpt C --rt RT_IN --out RT_OUT   dense.bin re-encoded (experts, draft vocab copied)
import argparse, json, math, os, shutil, struct, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

H, HC, LR, NH, NKV, HD, NROT, FF, NE, K = 2560, 4, 320, 24, 2, 256, 64, 640, 512, 10
EPS, THETA = 1e-6, 1e7
TAIL = int(os.environ.get("MTP_TAIL", "384"))   # session replays: loss / eval on the last TAIL cells (the model's own answer)
D = Path(os.environ.get("MTP_D", Path.home() / "work/rdna3/data/mtp"))
DEV = os.environ.get("MTP_DEV", "cuda")


# ---------------------------------------------------------------- weights
def rt_kinds(rt):
    return {l.split()[0]: l.split()[1] for l in open(Path(rt) / "dense.txt") if l.strip()}

def load_bf16(name, shape):
    raw = np.fromfile(D / "bf16/tensors" / f"{name}.bin", dtype=np.uint16)
    return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy())

def load_weights(rt):
    inv = {t["name"]: t["shape"] for t in json.load(open(D / "mtp-inventory.json"))["tensors"]}
    kinds = rt_kinds(rt)
    w = {}
    for full, shape in inv.items():
        short = full[len("mtp."):]
        short = short[len("layers.0."):] if short.startswith("layers.0.") else short
        if "experts." in short or "indexer" in short: continue
        t = load_bf16(full, shape)
        if kinds.get(short) == "f32": t = t.reshape(-1) + 1.0        # Gemma RMSNorm: the rt stores 1 + w
        w[short] = t
    gu = load_bf16("mtp.layers.0.mlp.experts.gate_up_proj", inv["mtp.layers.0.mlp.experts.gate_up_proj"])
    dn = load_bf16("mtp.layers.0.mlp.experts.down_proj", inv["mtp.layers.0.mlp.experts.down_proj"])
    return w, gu, dn


# ---------------------------------------------------------------- run-time quantization (--qat)
def q4_0(x):
    """ggml's reference Q4_0 over blocks of 32 along the last dim, dequantized: what --mtp-q4 makes of the Q8_0 projections
    and the draft head, and what the q4_0 expert pack holds."""
    shp = x.shape
    b = x.reshape(-1, 32).float()
    mx = b.gather(1, b.abs().argmax(1, keepdim=True))
    d = mx / -8.0
    q = torch.trunc(b * torch.where(d == 0, torch.zeros_like(d), 1.0 / d) + 8.5).clamp(0, 15)
    return ((q - 8.0) * d.half().float()).reshape(shp).to(x.dtype)

def q4_big(t, step=16):   # in place, in slices (the experts, the head)
    for i in range(0, t.shape[0], step): t[i:i + step] = q4_0(t[i:i + step].to(DEV)).to(t.device)
    return t


# ---------------------------------------------------------------- the draft layer (mtp.cpp record_front + record_rest)
def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w

def gr_read(R, p, pre, inject=True):
    """R (T, HC, H) -> mixed (T, H), inject (T, HC).  gr.hpp: per-stream norm, lo = silu(xn @ down.T / hc), gate = sigmoid(lo @ up.T)."""
    T = R.shape[0]
    xn = rms(R, p[pre + "hc_norm.weight"].view(HC, H))
    xf = xn.reshape(T, HC * H)
    lo = F.silu((xf @ p[pre + "input_mix_weight_down.weight"].T) / HC)
    gate = torch.sigmoid(lo @ p[pre + "input_mix_weight_up.weight"].T).view(T, HC, H)
    mixed = (xn * gate).mean(1)
    inj = xf @ p[pre + "block_inject_weight.weight"].T if inject else None
    return mixed, inj

def gr_write(R, y, inj):
    return R + y[:, None, :] * (2 * torch.sigmoid(inj / HC))[:, :, None]

def rope(x, pos):   # x (T, heads, HD), neox on the first NROT dims
    half = NROT // 2
    inv = 1.0 / (THETA ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos[:, None].float() * inv[None]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    a, b = x[..., :half], x[..., half:NROT]
    return torch.cat([a * c - b * s, b * c + a * s, x[..., NROT:]], -1)

class Mtp(torch.nn.Module):
    def __init__(self, w, gu, dn, embed, head, hnorm_stream=True):
        super().__init__()
        self.p = torch.nn.ParameterDict({k.replace(".", "__"): torch.nn.Parameter(v) for k, v in w.items()})
        self.register_buffer("gu", gu.to(torch.bfloat16), persistent=False)   # (NE, 2FF, H): gate rows then up rows
        self.register_buffer("dn", dn.to(torch.bfloat16), persistent=False)   # (NE, H, FF)
        self.register_buffer("embed", embed, persistent=False)                # (V, H) bf16
        self.register_buffer("head", head, persistent=False)                  # (V, H) bf16
        self.hnorm_stream = hnorm_stream
        self.qat = set()   # --qat: the tensors the server reads as Q4_0 (straight-through in training)

    def w(self):
        return {k.replace("__", "."): (v + (q4_0(v) - v).detach() if k in self.qat else v) for k, v in self.p.items()}

    def experts(self, x, ids, wts):
        out = torch.zeros_like(x)
        flat = ids.reshape(-1)
        tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(K)
        wf = wts.reshape(-1)
        order = torch.argsort(flat)
        flat, tok, wf = flat[order], tok[order], wf[order]
        cnt = torch.bincount(flat, minlength=NE).tolist()
        o = 0
        xb = x.to(torch.bfloat16)
        for e, c in enumerate(cnt):
            if c == 0: continue
            t = tok[o:o + c]
            h = xb[t] @ self.gu[e].T
            a = F.silu(h[:, :FF].float()) * h[:, FF:].float()
            y = (a.to(torch.bfloat16) @ self.dn[e].T).float() * wf[o:o + c, None]
            out.index_add_(0, t, y)
            o += c
        return out

    def forward(self, Rin, tok_next, pos, kv=None):
        """Rin (T, HC*H) final residual rows, tok_next (T,) the token at i+1, pos (T,) cell positions.
        kv: None = causal over these T cells; else (K1, V1) of the step-1 cells: row t sees K1[:t+1-?] (see step2)."""
        p = self.w()
        T = Rin.shape[0]
        e = rms(self.embed[tok_next].float(), p["pre_fc_norm_embedding.weight"]) @ p["fc_embedding.weight"].T
        if self.hnorm_stream:
            hn = rms(Rin.view(T, HC, H), p["pre_fc_norm_hidden.weight"].view(HC, H))
        else:
            hn = rms(Rin, p["pre_fc_norm_hidden.weight"]).view(T, HC, H)
        R = hn @ p["fc_hidden.weight"].T + e[:, None, :]
        mixed, inj = gr_read(R, p, "attn_hyper_connection.")
        qf = (mixed @ p["self_attn.q_proj.weight"].T).view(T, NH, 2 * HD)
        q, qg = qf[..., :HD], qf[..., HD:]
        q = rope(rms(q, p["self_attn.q_norm.weight"]), pos)
        k = rope(rms((mixed @ p["self_attn.k_proj.weight"].T).view(T, NKV, HD), p["self_attn.k_norm.weight"]), pos)
        v = (mixed @ p["self_attn.v_proj.weight"].T).view(T, NKV, HD)
        rep = NH // NKV
        if kv is None:
            kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            att = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kk.transpose(0, 1)[None], vv.transpose(0, 1)[None],
                                                 is_causal=True)[0].transpose(0, 1)
        else:   # step >= 2: row t (cell c = pos[t]) sees the step-1 cells < c and its own k/v at c
            K1, V1, cells = kv   # (n, NKV, HD), cell index of each K1 row
            kk, vv = K1.repeat_interleave(rep, 1), V1.repeat_interleave(rep, 1)
            s = torch.einsum("thd,nhd->htn", q, kk) / math.sqrt(HD)
            s = s.masked_fill(cells[None, None, :] >= pos[None, :, None], float("-inf"))
            s_self = torch.einsum("thd,thd->ht", q, k.repeat_interleave(rep, 1))[..., None] / math.sqrt(HD)
            pr = torch.softmax(torch.cat([s, s_self], -1), -1)
            att = torch.einsum("htn,nhd->thd", pr[..., :-1], vv) + pr[..., -1:].permute(1, 0, 2) * v.repeat_interleave(rep, 1)
        o = (att * torch.sigmoid(qg)).reshape(T, NH * HD) @ p["self_attn.o_proj.weight"].T
        R = gr_write(R, o, inj)
        mixed, inj2 = gr_read(R, p, "mlp_hyper_connection.")
        logits = mixed @ p["mlp.gate.weight"].T
        pr = torch.softmax(logits.float(), -1)
        wts, ids = pr.topk(K, -1)
        wts = wts / wts.sum(-1, keepdim=True).clamp_min(6.1e-5)
        y = self.experts(mixed, ids, wts)
        sh = (F.silu(mixed @ p["mlp.shared_expert.gate_proj.weight"].T) * (mixed @ p["mlp.shared_expert.up_proj.weight"].T)) \
            @ p["mlp.shared_expert.down_proj.weight"].T
        y = y + sh * torch.sigmoid(mixed @ p["mlp.shared_expert_gate.weight"].T)
        R = gr_write(R, y, inj2)
        sample, _ = gr_read(R, p, "hyper_connection_mixer.", inject=False)
        return sample, R.reshape(T, HC * H), (k, v)


# ---------------------------------------------------------------- data (generate.cpp STRATA_MTP_DUMP records)
def read_dump(spec):
    """comma-separated dumps -> sequences tagged with their file (toks, chunks, path); eval = the last file's tail."""
    return [(t, c, p) for p in spec.split(",") for t, c in read_dump_one(p)]

def read_dump_one(path):
    """-> list of (tokens int64 array, [(p0, T, offset of the bf16 rows)]) per sequence (a record with p0 == 0 starts one)."""
    seqs, cur = [], None
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        off = 0
        while off + 28 <= size:
            f.seek(off)
            hdr = f.read(28)
            assert hdr[:4] == b"MTPD", f"bad record at {off}"
            p0, T, n = struct.unpack("<qqq", hdr[4:])
            toks = np.frombuffer(f.read(4 * n), dtype=np.int32).astype(np.int64)
            roff = off + 28 + 4 * n
            if roff + T * HC * H * 2 > size: break
            if p0 == 0:
                cur = (toks, [])
                seqs.append(cur)
            if cur is not None and len(toks) == len(cur[0]) and p0 == sum(c[1] for c in cur[1]):
                cur[1].append((p0, T, roff))
            else:
                cur = None
            off = roff + T * HC * H * 2
    return [s for s in seqs if s[1]]

def seq_rows(path, chunks):
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    parts = [np.asarray(mm[o // 2: o // 2 + T * HC * H]).reshape(T, HC * H) for _, T, o in chunks]
    u = np.concatenate(parts)
    return torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))

def batch(path, seq, dev=DEV, max_len=2048):
    toks, chunks = seq[0], seq[1]
    R = seq_rows(seq[2] if len(seq) > 2 else path, chunks)
    n = min(R.shape[0], len(toks) - 2)   # cell i: R_i, token i+1, label i+2
    s = max(0, n - max_len)              # a long sequence (a session replay): its last max_len cells, where the answer is
    src = str(seq[2] if len(seq) > 2 else path)
    lo = max(0, n - s - TAIL) if "-sess" in src else 0   # a session replay (data/mtp/*-sess): only the answer's cells count, the session's own text is not this model's
    return (R[s:n].to(dev), torch.from_numpy(toks[s + 1:n + 1]).to(dev), torch.from_numpy(toks[s + 2:n + 2]).to(dev),
            torch.arange(s, n, device=dev), lo)


# ---------------------------------------------------------------- commands
def prep(a):
    sys.path.insert(0, str(Path.home() / "work/rdna3/engine/third_party/llama.cpp/gguf-py"))
    import gguf
    from gguf.quants import dequantize
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.gguf).parent.glob(Path(a.gguf).name.split("-00001")[0] + "*.gguf")):
        r = gguf.GGUFReader(str(f))
        for t in r.tensors:
            if t.name in ("token_embd.weight", "output.weight"):
                x = dequantize(t.data, t.tensor_type).reshape(-1, H)
                torch.save(torch.from_numpy(x).to(torch.bfloat16), out / ("embed.pt" if t.name.startswith("token") else "head.pt"))
                print(t.name, t.tensor_type.name, x.shape, flush=True)

def make_model(a):
    w, gu, dn = load_weights(a.rt)
    embed, head = torch.load(Path(a.w) / "embed.pt"), torch.load(Path(a.w) / "head.pt")
    m = Mtp(w, gu, dn, embed, head, hnorm_stream=not a.pooled).to(DEV)
    if getattr(a, "ckpt", None):
        m.p.load_state_dict(torch.load(a.ckpt), strict=False)
    if getattr(a, "qat", False):   # what the server computes with: Q4_0 projections (--mtp-q4 all), q4_0 experts, Q4_0 draft head
        kinds = rt_kinds(a.rt)
        m.qat = {k for k in m.p.keys() if kinds.get(k.replace("__", ".")) == "q8_0"}
        with torch.no_grad(): q4_big(m.gu); q4_big(m.dn); q4_big(m.head, 4096)
        print(f"qat: {len(m.qat)} projections, the experts and the head at Q4_0", flush=True)
    return m

def vocab_mask(a, V):
    if not a.vocab: return None
    ids = np.fromfile(a.vocab, dtype=np.int32)
    ids = ids[1:] if ids[0] == len(ids) - 1 else ids   # tolerate a count header
    m = torch.full((V,), float("-inf"), device=DEV); m[torch.from_numpy(ids.astype(np.int64)).to(DEV)] = 0
    return m

@torch.no_grad()
def evaluate(m, path, seqs, mask, steps=int(os.environ.get("MTP_EVAL_STEPS", "3"))):   # accuracy of draft steps 1..steps on the accepted path
    hit = [0] * steps; tot = [0] * steps
    for s in seqs:
        R, tn, lab, pos, lo = batch(path, s)
        sample, Rout, (k, v) = m(R, tn, pos)
        lg = sample.to(torch.bfloat16) @ m.head.T
        if mask is not None: lg = lg.float() + mask
        pred = lg.argmax(-1)
        hit[0] += (pred == lab)[lo:].sum().item(); tot[0] += len(lab) - lo
        # chain: step j row t = cell t + j, from the previous step's residual and the true token (accepted path)
        Rp, n = Rout, len(lab)
        for j in range(1, steps):
            if n - j <= 0: break
            toks = s[0][int(pos[0]):]                                        # (a long sequence starts at its tail window)
            tn_j = torch.from_numpy(toks[j + 1: j + 1 + n - j]).to(DEV)      # token at cell+1 = label of the previous step
            lab_j = torch.from_numpy(toks[j + 2: j + 2 + n - j]).to(DEV)
            pos_j = pos[: n - j] + j
            sample, Rp, _ = m(Rp[: n - j], tn_j[: len(lab_j)], pos_j[: len(lab_j)], kv=(k, v, pos))
            Rp = Rp
            lg = sample.to(torch.bfloat16) @ m.head.T
            if mask is not None: lg = lg.float() + mask
            hit[j] += (lg.argmax(-1)[: len(lab_j)] == lab_j)[lo:].sum().item(); tot[j] += max(len(lab_j) - lo, 0)
    return [h / max(t, 1) for h, t in zip(hit, tot)]

def cmd_eval(a):
    m = make_model(a); m.eval()
    seqs = read_dump(a.dump)
    ev = seqs[-a.n_eval:]
    print(f"{len(seqs)} sequences; eval on {len(ev)}:", ["%.4f" % x for x in evaluate(m, a.dump, ev, vocab_mask(a, m.head.shape[0]))])

def cmd_train(a):
    m = make_model(a)
    seqs = read_dump(a.dump)
    tr, ev = seqs[:-a.n_eval], seqs[-a.n_eval:]
    mask = vocab_mask(a, m.head.shape[0])
    tgt = (lambda y: y) if mask is None else (lambda y: torch.where(torch.isinf(mask[y]), -100, y))   # labels the draft vocab can't emit: ignored
    train_names = [k for k in m.p.keys() if not any(s in k for s in a.freeze.split(","))] if a.freeze else list(m.p.keys())
    for k, p in m.p.items(): p.requires_grad_(k in train_names)
    params = [m.p[k] for k in train_names]
    print(f"{len(tr)} train / {len(ev)} eval sequences; training {len(params)} tensors, "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M params", flush=True)
    print("base:", ["%.4f" % x for x in evaluate(m, a.dump, ev, mask)], flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    rng = np.random.default_rng(0)
    step, steps = 0, a.steps
    while step < steps:
        for i in rng.permutation(len(tr)):
            R, tn, lab, pos, lo = batch(a.dump, tr[i])
            def tgl(y, _lo=lo):   # the draft vocab's labels; cells before the answer of a long sequence are ignored
                y = tgt(y).clone(); y[:_lo] = -100; return y
            lr = a.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0)))
            for g in opt.param_groups: g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sample, Rout, (k, v) = m(R, tn, pos)
                loss = F.cross_entropy(((sample @ m.head.T).float() + (mask if mask is not None else 0)), tgl(lab))
                # steps 2..chain on the accepted path, each fed the previous step's own residual (FastMTP), weight chain_w^(j)
                Rp, n, toks = Rout, len(lab), tr[i][0][int(pos[0]):]
                for j in range(1, a.chain):
                    if n - j <= 0: break
                    tn_j = torch.from_numpy(toks[j + 1: j + 1 + n - j]).to(DEV)
                    lab_j = torch.from_numpy(toks[j + 2: j + 2 + n - j]).to(DEV)
                    sj, Rp, _ = m(Rp[: n - j], tn_j[: len(lab_j)], pos[: n - j][: len(lab_j)] + j, kv=(k.detach(), v.detach(), pos))
                    loss = loss + a.chain_w ** j * F.cross_entropy(((sj @ m.head.T).float() + (mask if mask is not None else 0)),
                                                                   tgl(lab_j))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 50 == 0: print(f"step {step} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
            if step % a.eval_every == 0 or step == steps:
                print(f"step {step} eval:", ["%.4f" % x for x in evaluate(m, a.dump, ev, mask)], flush=True)
                torch.save({k: v.detach().cpu() for k, v in m.p.items()}, a.out)
            if step >= steps: break

def q8_0(x):   # ggml reference Q8_0 (as tools/mtp_rt.py)
    x = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(x).max(1) / 127.0
    q = np.round(x / np.where(d == 0, 1, d)[:, None]).clip(-127, 127).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()

def cmd_export(a):
    sd = {k.replace("__", "."): v.float() for k, v in torch.load(a.ckpt).items()}
    src, out = Path(a.rt), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob = open(src / "dense.bin", "rb").read()
    lines, off, changed = [], 0, 0
    with open(out / "dense.bin", "wb") as f:
        for l in open(src / "dense.txt"):
            if not l.strip(): continue
            name, kind, rows, cols, o, nb = l.split(); o, nb = int(o), int(nb)
            raw = blob[o:o + nb]
            if name in sd:
                t = sd[name].numpy()
                raw = (t.astype(np.float32).tobytes() if kind == "f32" else q8_0(t) if kind == "q8_0"
                       else (t.astype(np.float32).view(np.uint32) + 0x7FFF + ((t.astype(np.float32).view(np.uint32) >> 16) & 1) >> 16).astype(np.uint16).tobytes())
                assert len(raw) == nb, (name, len(raw), nb)
                changed += 1
            pad = (-off) % 256; f.write(b"\0" * pad); off += pad
            f.write(raw); lines.append(f"{name} {kind} {rows} {cols} {off} {nb}"); off += nb
    (out / "dense.txt").write_text("\n".join(lines) + "\n")
    for n in ("experts.bin", "experts.txt", "draft_vocab.bin"):
        if (src / n).exists() and not (out / n).exists(): os.link(src / n, out / n)
    print(f"{changed} tensors re-encoded -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--gguf"); ap.add_argument("--out"); ap.add_argument("--w", default=str(D / "train"))
    ap.add_argument("--rt", default=str(D / "rt-q4_0")); ap.add_argument("--dump"); ap.add_argument("--ckpt")
    ap.add_argument("--vocab"); ap.add_argument("--pooled", action="store_true"); ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--freeze", default="")
    ap.add_argument("--chain", type=int, default=1); ap.add_argument("--chain-w", type=float, default=0.5)
    ap.add_argument("--qat", action="store_true", help="train / eval with the server's Q4_0 projections, experts and head (straight-through)")
    a = ap.parse_args()
    {"prep": prep, "eval": cmd_eval, "train": cmd_train, "export": cmd_export}[a.cmd](a)
