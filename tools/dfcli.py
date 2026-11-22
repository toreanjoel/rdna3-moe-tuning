#!/usr/bin/env python3
# dfcli.py — one CLI (non-serve) generation with a bench run's exact engine args/env plus extra flags; prints the engine's speed/draft lines
# usage (on the box, server stopped): python3 tools/dfcli.py bench/<run>/iq3-1gpu/config.json <tokens.txt> <max_new> [extra engine args ...]
import json, os, subprocess, sys
cfg = json.load(open(sys.argv[1]))
drop_val = {"--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache", "--host", "--port"}
args, a = [], cfg["args"]
i = 0
while i < len(a):
    if a[i] in drop_val: i += 2; continue
    if a[i] == "--serve": i += 1; continue
    args.append(a[i]); i += 1
env = dict(os.environ, **cfg.get("env", {}))
env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
cmd = [cfg["exe"]] + args + ["--tokens-file", sys.argv[2], "--max-new", sys.argv[3]] + sys.argv[4:]
p = subprocess.run(cmd, cwd=cfg["cwd"], env=env, capture_output=True, text=True)
out = p.stdout + p.stderr
keep = ("decode ", "rounds", "tok/s", "accept", "draft", "dflash", "window", "error", "Error", "spec", "generated")
if p.returncode: print(out[-2500:])
for line in out.splitlines():
    if any(k in line for k in keep) and "captured" not in line: print(line[:300])
print("exit", p.returncode)
