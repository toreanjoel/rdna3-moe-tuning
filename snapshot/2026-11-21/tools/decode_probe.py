#!/usr/bin/env python3
"""decode_probe.py — measure Strata decode tok/s at a fixed long context.

Sends the same long prompt + short instruction N times to the local Strata
server, then reads the engine's own /v1/status last_timings. Same prompt each
time, so prompt processing is a cache hit and the number is pure generation.
Read-only: never touches configs, only HTTP + GET /v1/status.

usage: ./decode_probe.py [reps] [max_tokens] [prompt_kib]
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080"

BLOCK = (
    "The RX 7900 XTX runs the decode step in three parts: attention over the "
    "resident KV, the router for each MoE layer, and the expert weights that "
    "live in video memory. When an expert is not cached, Strata fetches the row "
    "over PCIe or computes it on the CPU while the GPU continues.\n"
    "def measure(profile, reps):\n"
    "    out = []\n"
    "    for i in range(reps):\n"
    "        out.append(run_decode(profile, tokens=300))\n"
    "    return statistics.median(out)\n"
    "Notes: keep the prefix identical between runs so the conversation cache "
    "reuses it; vary only one setting at a time; write every run down.\n"
)


def post(path, payload, timeout=1200):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def status():
    with urllib.request.urlopen(URL + "/v1/status", timeout=30) as r:
        return json.load(r)


def wait_idle(limit=30):
    """Only measure when the server is otherwise idle (concurrency is 1)."""
    end = time.time() + limit
    while time.time() < end:
        if not status().get("activity", {}).get("in_flight"):
            return True
        time.sleep(1)
    return False


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    kib = int(sys.argv[3]) if len(sys.argv) > 3 else 112
    label = " ".join(sys.argv[4:]) or "probe"

    prompt = BLOCK * max(1, int(kib * 1024 // len(BLOCK)))
    msgs = [
        {"role": "system", "content": "Answer with a short list. No preamble."},
        {"role": "user", "content": prompt + "\n\nList the five biggest decode "
         "bottlenecks for this machine, one line each."},
    ]

    rates, accs = [], []
    for i in range(reps):
        wait_idle()
        t0 = time.time()
        post("/v1/chat/completions", {"messages": msgs, "max_tokens": max_tokens,
                                      "temperature": 0.0, "stream": False})
        wall = time.time() - t0
        lt = status().get("last_timings", {})
        tps = lt.get("predicted_per_second", 0.0)
        dn, da = lt.get("draft_n", 0), lt.get("draft_n_accepted", 0)
        rates.append(tps)
        accs.append(100.0 * da / dn if dn else 0.0)
        print(f"  rep{i+1}: {tps:6.1f} tok/s  gen {lt.get('predicted_n')} in "
              f"{lt.get('predicted_ms', 0)/1000:.1f}s  cache {lt.get('cache_n')} "
              f"prompt {lt.get('prompt_n')}  draft {da}/{dn} "
              f"({accs[-1]:.0f}%)  wall {wall:.1f}s", flush=True)

    s = sorted(rates)
    med = s[len(s) // 2]
    print(f"RESULT [{label}] median {med:.1f} tok/s  min {min(rates):.1f}  "
          f"max {max(rates):.1f}  mean_acc {sum(accs)/len(accs):.0f}%  reps {reps}")


if __name__ == "__main__":
    main()
