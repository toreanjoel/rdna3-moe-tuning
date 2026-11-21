#!/usr/bin/env python3
"""decode_watts.py — GPU watts during decode only, plus tok/s for the same window.

Answers "does this clock profile stay under my power budget while generating?".
Warm-up request first (prompt cache), then one long generation sampled at 5 Hz;
samples are kept only while the server reports a request in flight, so prompt
processing and idle are excluded. Needs no root (hwmon power1_average is readable).

usage: decode_watts.py <max_tokens> [prompt_kib] [power_cap_w_to_check] [label]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080"
HW = "/sys/class/drm/card1/device/hwmon/hwmon5/power1_average"

BLOCK = open(__file__).read()  # stable filler text; same bytes every run


def status():
    with urllib.request.urlopen(URL + "/v1/status", timeout=30) as r:
        return json.load(r)


def post(payload, timeout=1800):
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    kib = int(sys.argv[2]) if len(sys.argv) > 2 else 112
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    label = " ".join(sys.argv[4:]) or "profile"

    prompt = BLOCK * max(1, int(kib * 1024 // len(BLOCK)))
    msgs = [{"role": "system", "content": "Answer in long numbered detail."},
            {"role": "user", "content": prompt + "\n\nExplain, in detail, how a GPU "
             "expert cache works, then list trade-offs."}]

    post({"messages": msgs, "max_tokens": 400, "temperature": 0.0})  # warm

    watts, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                with open(HW) as f:
                    w = int(f.read().strip()) / 1e6
                if status().get("activity", {}).get("in_flight"):
                    watts.append(w)
            except Exception:
                pass
            time.sleep(0.2)

    t = threading.Thread(target=sample)
    t.start()
    post({"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0})
    stop.set()
    t.join()

    lt = status().get("last_timings", {})
    mean_w = statistics.mean(watts) if watts else 0.0
    p95 = sorted(watts)[int(len(watts) * 0.95) - 1] if watts else 0.0
    print(f"WATTS [{label}] decode {lt.get('predicted_per_second', 0):.1f} tok/s "
          f"({lt.get('predicted_n')} tok) | mean {mean_w:.0f} W p95 {p95:.0f} W "
          f"max {max(watts):.0f} W | n={len(watts)} | budget {budget} W -> "
          f"{'OK' if max(watts) <= budget else 'OVER'}")


if __name__ == "__main__":
    main()
