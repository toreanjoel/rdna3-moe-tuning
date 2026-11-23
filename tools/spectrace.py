# spectrace.py — summarize a STRATA_SPEC_TRACE file ("T a sfx ms p1 p2 ..." per round): tokens/round, per-depth live acceptance, window mix, ms/round, and acceptance by draft-probability bucket
import sys
from collections import Counter, defaultdict

rows = []
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 4: continue
    try:
        T, a, sfx, ms = int(f[0]), int(f[1]), int(f[2]), float(f[3])
    except ValueError:
        continue
    rows.append((T, a, sfx, ms, [float(x) for x in f[4:]]))
n = len(rows)
if not n: sys.exit("no rows")
print(f"rounds {n}  tokens/round {sum(r[1] + 1 for r in rows) / n:.3f}  ms/round {sum(r[3] for r in rows) / n:.2f}  "
      f"tok/s {sum(r[1] + 1 for r in rows) / sum(r[3] for r in rows) * 1000:.1f}")
print("window rows T:", dict(sorted(Counter(r[0] for r in rows).items())))
print("accepted a:  ", dict(sorted(Counter(r[1] for r in rows).items())))
print("suffix rounds:", sum(1 for r in rows if r[2] > 0))
# live depth acceptance: P(draft d accepted | drafts 1..d-1 accepted and draft d was in the window)
for d in range(1, 9):
    tried = [r for r in rows if r[0] - 1 >= d and r[1] >= d - 1]
    if not tried: break
    ok = sum(1 for r in tried if r[1] >= d)
    print(f"depth {d}: offered {len(tried)}, accepted {ok / len(tried):.3f}")
# acceptance vs the draft's own probability (first draft only, MTP rounds)
b = defaultdict(lambda: [0, 0])
for T, a, sfx, ms, p in rows:
    if sfx or not p or T < 2: continue
    k = min(int(p[0] * 10), 9)
    b[k][0] += 1; b[k][1] += a >= 1
print("draft-1 p bucket: n, accepted")
for k in sorted(b):
    print(f"  {k / 10:.1f}-{(k + 1) / 10:.1f}: {b[k][0]:6d} {b[k][1] / b[k][0]:.3f}")
# rounds where the window was cut by min-p (T-1 < max drafted probabilities)
cut = sum(1 for T, a, sfx, ms, p in rows if not sfx and len(p) > T - 1)
print(f"MTP rounds with drafts dropped by the window rule: {cut} / {sum(1 for r in rows if not r[2])}")
by_T = defaultdict(list)
for r in rows: by_T[r[0]].append(r[3])
print("ms by window rows:", {T: round(sum(v) / len(v), 2) for T, v in sorted(by_T.items())})
