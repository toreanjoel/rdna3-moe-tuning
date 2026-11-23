# spectrace.py — summarize a STRATA_SPEC_TRACE file from the server ("T a sfx ms thr p1 .. pn" per round; --cli: the CLI's "T a sfx ms p1 ..") : tokens/round, zero-accept split, per-position offered / probability / accepted, head calibration, ms by window rows
# usage: spectrace.py TRACE [--cli] [--acc 0.819,0.762,0.723]   (--acc: offline per-step head accuracy, for the predicted numbers)
import sys
from collections import Counter, defaultdict

cli = "--cli" in sys.argv
acc = [float(x) for x in sys.argv[sys.argv.index("--acc") + 1].split(",")] if "--acc" in sys.argv else [0.819, 0.762, 0.723]
rows = []
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 4: continue
    try:
        T, a, sfx, ms = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        thr = 0.0 if cli else float(f[4])
        p = [float(x) for x in f[(4 if cli else 5):]]
    except (ValueError, IndexError):
        continue
    rows.append((T, a, sfx, ms, p, thr))
n = len(rows)
if not n: sys.exit("no rows")
tok, tms = sum(r[1] + 1 for r in rows), sum(r[3] for r in rows)
print(f"rounds {n}  tokens/round {tok / n:.3f}  ms/round {tms / n:.2f}  tok/s {tok / tms * 1000:.1f}  mean window rows {sum(r[0] for r in rows) / n:.2f}")
print("window rows T:", dict(sorted(Counter(r[0] for r in rows).items())), " accepted a:", dict(sorted(Counter(r[1] for r in rows).items())))
z = [r for r in rows if r[1] == 0]
z0 = sum(1 for r in z if r[0] == 1)
print(f"zero-accept rounds {len(z)} ({len(z) / n:.1%} of rounds): no draft offered {z0} ({z0 / n:.1%}), offered but rejected {len(z) - z0} ({(len(z) - z0) / n:.1%})")
# per draft position d (window row d + 1): drafted by the head, offered (in the window), reached (drafts before it held), accepted
print("pos  drafted  mean_p  offered  mean_p  reached  accepted  acc|reached  acc|all rounds")
for d in range(1, 8):
    drafted = [r for r in rows if len(r[4]) >= d]
    if not drafted: break
    off = [r for r in rows if r[0] - 1 >= d]
    reach = [r for r in off if r[1] >= d - 1]
    ok = sum(1 for r in reach if r[1] >= d)
    mp = lambda rs: sum(r[4][d - 1] for r in rs if len(r[4]) >= d) / max(1, sum(1 for r in rs if len(r[4]) >= d))
    print(f"{d:3d}  {len(drafted):7d}  {mp(drafted):.3f}  {len(off):7d}  {mp(off):.3f}  {len(reach):7d}  {ok:8d}  {ok / max(1, len(reach)):11.3f}  {ok / n:.3f}")
# what the offline accuracy predicts if every draft were offered
e, q = 1.0, 1.0
for x in acc: q *= x; e += q
print(f"offline head accuracy {acc} predicts: zero-accept {1 - acc[0]:.1%}, {e:.2f} tokens/round with every draft offered")
# calibration: acceptance of a reached draft vs the head's probability, all positions
b = defaultdict(lambda: [0, 0])
for T, a, sfx, ms, p, thr in rows:
    if sfx: continue
    for d in range(1, T):
        if a < d - 1 or len(p) < d: break
        k = min(int(p[d - 1] * 10), 9); b[k][0] += 1; b[k][1] += a >= d
print("head p bucket (reached drafts): n, accepted")
for k in sorted(b): print(f"  {k / 10:.1f}-{(k + 1) / 10:.1f}: {b[k][0]:6d} {b[k][1] / b[k][0]:.3f}")
# the first draft the head made but the window rule left out: its probability
lo = [r[4][r[0] - 1] for r in rows if not r[2] and len(r[4]) > r[0] - 1]
if lo:
    h = Counter(min(int(x * 10), 9) for x in lo)
    print(f"rounds cut by the window rule: {len(lo)} / {n}; p of the cut draft:", {f"{k / 10:.1f}": v for k, v in sorted(h.items())})
by_T = defaultdict(list)
for r in rows: by_T[r[0]].append(r[3])
med = lambda v: sorted(v)[len(v) // 2]
print("round ms by window rows (median, n):", {T: (round(med(v), 2), len(v)) for T, v in sorted(by_T.items())})
if any(r[5] > 0 for r in rows): print(f"mean CUMX row-2 threshold {sum(r[5] for r in rows) / n:.3f}")
