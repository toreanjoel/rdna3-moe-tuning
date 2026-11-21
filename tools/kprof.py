#!/usr/bin/env python3
# kprof.py — summarise a kprof.sh trace: per-kernel exclusive GPU time, count, avg inside the chat windows; idle gaps; per token
# Graph-replayed kernels report start times that overlap their predecessor, so a kernel's time here is exclusive:
# end - max(start, previous end), in end order (decode is one in-order stream); idle = start - previous end when > 0.
import csv, glob, json, re, sys
from collections import defaultdict

d = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
win = json.load(open(d + "/windows.json"))
spans = [(w["t0"], w["t1"]) for w in win]

def name(n):
    n = re.sub(r"\(anonymous namespace\)::", "", n)
    n = re.sub(r"^void ", "", n).replace("strata::kernels::", "")
    return n.split("(")[0][:80]

rows = []
for f in glob.glob(d + "/trace/**/*kernel_trace.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        for i, (a, b) in enumerate(spans):
            if a <= s and e <= b:
                rows.append((i, e, s, name(r["Kernel_Name"]), r["Queue_Id"]))
                break
if not rows:
    sys.exit("no kernels inside the windows (clock mismatch?)")
rows.sort()

agg = defaultdict(lambda: [0, 0, 0])      # name -> [exclusive ns, count, raw ns]
toks = sum(w["tokens"] for w in win)
span = sum(b - a for a, b in spans)
idle, gaps, queues = 0, [], defaultdict(int)
cur, prev = -1, 0
for i, e, s, n, q in rows:
    if i != cur:
        cur, prev = i, s                    # first kernel of a window
    if s > prev:
        idle += s - prev; gaps.append(s - prev)
    agg[n][0] += max(e - max(s, prev), 0); agg[n][1] += 1; agg[n][2] += e - s
    queues[q] += 1
    prev = max(prev, e)

busy = sum(v[0] for v in agg.values())
n_all = sum(v[1] for v in agg.values())
print(f"windows {len(win)}  tokens {toks}  wall {span/1e9:.2f} s -> {toks/(span/1e9):.1f} tok/s (profiled)")
print(f"per token: wall {span/1e6/toks:.3f} ms  GPU busy {busy/1e6/toks:.3f} ms  idle gaps {idle/1e6/toks:.3f} ms  "
      f"launches {n_all/toks:.0f}  dispatches per queue {dict(queues)}")
gaps.sort()
if gaps:
    print(f"gaps: n/tok {len(gaps)/toks:.0f}  median {gaps[len(gaps)//2]/1e3:.1f} us  p90 {gaps[int(len(gaps)*.9)]/1e3:.1f} us  "
          f">100us: {sum(g for g in gaps if g > 1e5)/1e6/toks:.3f} ms/tok")
print(f"{'ms/tok':>7} {'%':>5} {'n/tok':>6} {'excl us':>7} {'raw us':>7}  kernel")
for n, (t, c, raw) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
    print(f"{t/1e6/toks:7.3f} {100*t/busy:5.1f} {c/toks:6.1f} {t/c/1e3:7.1f} {raw/c/1e3:7.1f}  {n}")
