#!/usr/bin/env python3
# sessbench.py — replay session requests (tools/sessions.py jsonl) against the running server, greedy: decode tok/s, prompt tok/s and tokens per request
# usage (engine dir, venv): .venv/bin/python <repo>/tools/sessbench.py REQUESTS.jsonl [port] [max_tokens] [limit]
import json, statistics, sys
sys.path.insert(0, "rdna3")
import bench
src = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
maxt = int(sys.argv[3]) if len(sys.argv) > 3 else 256
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1 << 30
model = json.load(open("rdna3/configs/iq3-1gpu.json")).get("model_name", "m")
dec, pre, ntok, fail = [], [], 0, 0
for i, line in enumerate(open(src)):
    if i >= limit: break
    r = json.loads(line)
    body = bench.payload(model, "", maxt); body["messages"] = r["messages"]
    try: v = bench.request(port, body)
    except Exception as e:
        fail += 1; print(f"{r['id']}: failed {str(e)[:120]}", flush=True); continue
    t = v.get("timings", {})
    if t.get("predicted_n", 0) >= 16: dec.append(t.get("predicted_per_second", 0))
    pre.append(t.get("prompt_per_second", 0)); ntok += t.get("predicted_n", 0)
print(f"mean {statistics.mean(dec) if dec else 0:.1f} tok/s decode over {len(dec)} requests ({ntok} tokens, prompt {statistics.mean(pre) if pre else 0:.0f} tok/s, {fail} failed)")
