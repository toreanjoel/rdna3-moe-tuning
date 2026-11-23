# kseq.py — print a run of consecutive kernel names (dispatch order) from a rocprofv3 kernel_trace.csv; usage: python3 - DIR START COUNT < kseq.py
import csv, glob, re, sys
d, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
f = glob.glob(d + "/trace/*kernel_trace.csv")[0]
rows = list(csv.DictReader(open(f)))
key = "Dispatch_Id" if "Dispatch_Id" in rows[0] else list(rows[0])[0]
rows.sort(key=lambda r: int(r[key]))
def short(n):
    n = re.sub(r"strata::kernels::|\(anonymous namespace\)::|^void ", "", n); n = re.sub(r"\(.*", "", n)
    return n[:70]
print(len(rows), "dispatches")
for i, r in enumerate(rows[start:start + count]):
    print(start + i, short(r["Kernel_Name"]), r.get("Grid_Size_X", ""), r.get("Workgroup_Size_X", ""))
