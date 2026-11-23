#!/usr/bin/env python3
# chatbench.py — decode tok/s and draft acceptance on varied short chat prompts (code, prose, Q&A, list) against the running server, greedy, 384 tokens
# usage (engine dir, venv): [CHATOUT=answers.json] .venv/bin/python <repo>/tools/chatbench.py [port] [rounds]    (CHATOUT: the answers, for tools/chatcmp.py)
import json, os, statistics, sys
sys.path.insert(0, "rdna3")
import bench
PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat, with tests.",
    "Explain how TCP congestion control works, including slow start, congestion avoidance and fast recovery.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Give a step-by-step plan to migrate a PostgreSQL database to a new server with minimal downtime.",
    "Refactor this JavaScript into modern async/await style and explain each change: function load(cb){fetch('/a').then(r=>r.json()).then(d=>{fetch('/b?id='+d.id).then(r=>r.json()).then(e=>cb(null,e)).catch(cb)}).catch(cb)}",
    "Compare Rust and Go for writing a high-throughput network service. Use a table and then a summary.",
]
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
model = json.load(open("rdna3/configs/iq3-1gpu.json")).get("model_name", "m")
per, texts = [], []
for r in range(rounds):
    for i, p in enumerate(PROMPTS):
        body = bench.payload(model, p, 384)
        v = bench.request(port, body)
        t = v.get("timings", {})
        per.append(t.get("predicted_per_second", 0))
        if r == 0: texts.append(v.get("content", ""))
        if r == rounds - 1: print(f"prompt {i}: {t.get('predicted_per_second', 0):6.1f} tok/s  {t.get('predicted_n', 0)} tokens")
print(f"mean {statistics.mean(per):.1f} tok/s over {len(per)} requests")
if os.environ.get("CHATOUT"): json.dump(texts, open(os.environ["CHATOUT"], "w"))
