#!/usr/bin/env python3
# mtp_files.py — replay rows from source files already on the box (code, docs): each row asks for a file and holds its text as the assistant's answer, so the labelled MTP training set (mtp_train.py label) covers real code without generating it
# usage: mtp_files.py OUT.jsonl CHARS ROOT [ROOT ...]     CHARS = total budget, split evenly over the roots; one chunk (<= 24,000 chars) a file, files picked in a fixed shuffled order
import hashlib, json, os, random, sys

EXT = {".c", ".h", ".cpp", ".hpp", ".cc", ".cu", ".py", ".md", ".sh", ".txt", ".cmake", ".js", ".ts", ".rs", ".go", ".java", ".toml", ".yaml", ".yml", ".html", ".css", ".sql", ".swift", ".kt", ".rb"}
SKIP = {".git", "node_modules", "__pycache__", "third_party", "data", "bench", "logs", "models", "build", "build-hip", "bin", ".venv", "tests_data"}
ASK = ["Write the file `{p}`.", "Show me the full contents of `{p}`.", "Create `{n}` for this project.", "I need `{p}`. Write it out in full.", "Implement `{n}`."]

def files(root):
    for d, sub, names in os.walk(root):
        sub[:] = sorted(x for x in sub if x not in SKIP and not x.startswith("build"))
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in EXT: yield os.path.join(d, n)

out, budget, roots = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rows = []
for root in roots:
    fs = list(files(os.path.expanduser(root)))
    random.Random(0).shuffle(fs)
    used = 0
    for f in fs:
        if used >= budget // len(roots): break
        try: sz = os.path.getsize(f)
        except OSError: continue
        if sz < 1500 or sz > 400000: continue
        t = open(f, errors="ignore").read()
        if len(t) > 24000: t = t[:24000].rsplit("\n", 1)[0] + "\n"
        if t.count("\n") < 20 or max(len(l) for l in t.split("\n")) > 2000: continue   # not minified / generated blobs
        rel = os.path.relpath(f, os.path.expanduser(root))
        k = int(hashlib.sha1(rel.encode()).hexdigest(), 16)
        rows.append({"id": f"f{k % 10**9}", "messages": [{"role": "user", "content": ASK[k % len(ASK)].format(p=rel, n=os.path.basename(rel))}], "content": t})
        used += len(t)
    print(f"{root}: {used / 1e6:.1f}M chars", flush=True)
random.Random(1).shuffle(rows)
with open(out, "w") as fo:
    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} rows, {sum(len(r['content']) for r in rows) / 1e6:.1f}M chars -> {out}")
