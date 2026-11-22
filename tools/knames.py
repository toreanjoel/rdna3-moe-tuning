#!/usr/bin/env python3
# knames.py — kernel launch counts by short name from a rocprofv3 kernel_trace.csv (names/counts only; durations in graphs are unreliable)
# usage: python3 tools/knames.py logs/kprof-k4 [regex] [top N]
import csv, collections, glob, re, sys
d = sys.argv[1]; pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"gu|down|expdec|swiglu|s26|s27|expert")
c = collections.Counter()
for f in glob.glob(d + "/trace/*kernel_trace.csv"):
    for r in csv.DictReader(open(f)):
        n = r.get("Kernel_Name", "")
        m = re.search(r"kernels::(?:\(anonymous namespace\)::)?(\w+)(<[^(]*?>)?", n)
        k = (m.group(1) + (m.group(2) or ""))[:90] if m else n[:90]
        if pat.search(k): c[k] += 1
for k, v in c.most_common(int(sys.argv[3]) if len(sys.argv) > 3 else 20): print(v, k)
