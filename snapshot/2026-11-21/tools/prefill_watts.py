#!/usr/bin/env python3
"""prefill_watts.py — GPU watts and tok/s while READING a prompt (no cache reuse).

Decode stays under the 200 W budget; prompt reading is what pushes the card to
300+ W, so this is the check that decides MAX_SCLK. The prompt text is rebuilt
from /usr/share/dict words or urandom-derived hex each run so the conversation
cache never helps. max_tokens=1: only the prompt read is measured.

usage: prefill_watts.py [target_tokens] [power_budget_w] [label]
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080"
HW = "/sys/class/drm/card1/device/hwmon/hwmon5/power1_average"


def status():
    with urllib.request.urlopen(URL + "/v1/status", timeout=30) as r:
        return json.load(r)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    label = " ".join(sys.argv[3:]) or "profile"

    rnd = random.Random(os.urandom(8))
    unit = " ".join(f"tok{rnd.randrange(10**9):09d}" for _ in range(200))
    reps = max(1, int(target * 4.2 // len(unit)))          # ~4.2 chars per token
    prompt = (unit * reps)[:int(target * 4.2)]
    msgs = [{"role": "user", "content": prompt + "\nReply with one word."}]

    watts, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                w = int(open(HW).read().strip()) / 1e6
                if status().get("activity", {}).get("in_flight"):
                    watts.append(w)
            except Exception:
                pass
            time.sleep(0.2)

    req = {"messages": msgs, "max_tokens": 1, "temperature": 0.0}
    t = threading.Thread(target=sample)
    t.start()
    try:
        body = json.dumps(req).encode()
        r = urllib.request.urlopen(urllib.request.Request(
            URL + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}), timeout=3600)
        r.read()
    finally:
        stop.set()
        t.join()

    lt = status().get("last_timings", {})
    mean_w = statistics.mean(watts) if watts else 0.0
    peak = max(watts) if watts else 0.0
    over = sum(1 for w in watts if w > budget) * 0.2
    print(f"PREFILL [{label}] read {lt.get('prompt_n')} tok at "
          f"{lt.get('prompt_per_second', 0):.0f} tok/s | mean {mean_w:.0f} W "
          f"peak {peak:.0f} W | n={len(watts)} | budget {budget} W -> "
          f"{'OK' if peak <= budget else 'OVER'} | {over:.1f} s above {budget} W")


if __name__ == "__main__":
    main()
