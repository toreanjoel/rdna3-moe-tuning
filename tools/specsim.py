#!/usr/bin/env python3
# specsim.py — replay STRATA_SPEC_TRACE rounds under other window rules, offline: what the cumulative rule would keep at other scales / row costs / caps, with the head's raw probability or its measured acceptance, against the oracle (only the drafts that held); best fed a permissive trace (tools/wide_ab.sh arm s8x)
# usage: specsim.py TRACE [TRACE ...] [--rowms 1.1,2.0,2.9,3.6] [--cap 8]    ("T a sfx ms thr p1 .. pn" per round; lookup rounds are left as they ran)
#        a rule that wants rows the trace never verified is cut to what was verified (the "cut short" column): feed a wider trace then
import sys
from collections import defaultdict

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
rowms = [float(x) for x in opt("--rowms", "1.1,2.0,2.9,3.6").split(",")]
cap = int(opt("--cap", "8"))
rows = []
for path in args:
    for line in open(path):
        f = line.split()
        try: rows.append((int(f[0]), int(f[1]), int(f[2]), float(f[3]), [float(x) for x in f[5:]]))
        except (ValueError, IndexError): continue
if not rows: sys.exit("no rounds")
rm = lambda k: rowms[min(k, len(rowms)) - 1]            # ms that row k + 1 adds (draft k)
by = defaultdict(list)
for T, a, sfx, ms, p in rows:
    if not sfx: by[T].append(ms)
med = {T: sum(v) / len(v) for T, v in by.items() if len(v) >= 5}   # means, so the totals match the run
base = med.get(1) or (min(med.items())[1] - sum(rm(k) for k in range(1, min(med.items())[0])))
def cost(T):                                             # ms of a round of T rows: measured mean, else the last one + row costs
    if T in med: return med[T]
    lo = max((t for t in med if t < T), default=None)
    return (med[lo] if lo else base) + sum(rm(k) for k in range(lo or 1, T))
# measured acceptance of a reached draft by its probability (0.1 buckets) = the calibration a fixed table would give
hit, cnt = [0] * 10, [0] * 10
for T, a, sfx, ms, p in rows:
    if sfx: continue
    for k in range(1, min(T - 1, a + 1, len(p)) + 1):   # draft k was reached when the k - 1 before it held
        b = min(9, int(p[k - 1] * 10)); cnt[b] += 1; hit[b] += k <= a
cal = [(hit[b] + 1) / (cnt[b] + 2) for b in range(10)]

def run(scale, calibrated, cap_rows):
    rate = 0.13
    for _ in range(4):                                   # the rule's tokens-per-ms is its own result: iterate to it
        tok = ms_sum = short = 0
        for T, a, sfx, ms, p in rows:
            if sfx: tok += a + 1; ms_sum += ms; continue
            cum, m = 1.0, 0
            for k in range(1, min(len(p), cap_rows - 1) + 1):
                cum *= cal[min(9, int(p[k - 1] * 10))] if calibrated else p[k - 1]
                if cum < rate * rm(k) * scale: break
                m = k
            if m > T - 1: short += 1; m = T - 1
            tok += 1 + min(a, m); ms_sum += cost(m + 1)
        rate = tok / ms_sum
    return tok / len(rows), ms_sum / len(rows), 1000 * rate, short / len(rows)

n = len(rows)
tok, tms = sum(r[1] + 1 for r in rows), sum(r[3] for r in rows)
print(f"{n} rounds as run: {tok / n:.3f} tokens, {tms / n:.2f} ms a round, {1000 * tok / tms:.1f} tok/s; mean ms by rows: " + ", ".join(f"{T}: {v:.2f}" for T, v in sorted(med.items())))
print("acceptance of a reached draft by head p (0.1 buckets): " + " ".join(f"{c:.2f}" for c in cal))
o_tok = sum(r[1] + 1 for r in rows); o_ms = sum(r[3] if r[2] else cost(r[1] + 1) for r in rows)
print(f"oracle (only the drafts that held, of those verified): {o_tok / n:.3f} tokens, {o_ms / n:.2f} ms, {1000 * o_tok / o_ms:.1f} tok/s")
print("rule                       tokens     ms   tok/s  cut short")
for calibrated in (False, True):
    for c in sorted({4, cap}):
        for scale in (0.5, 0.7, 1.0, 1.4, 2.0):
            t, m, r, s = run(scale, calibrated, c)
            print(f"{'measured' if calibrated else 'raw p   '} cap {c} scale {scale:<4}  {t:6.3f} {m:6.2f}  {r:6.1f}  {s:6.1%}")
