#!/usr/bin/env bash
# kprof.sh — per-kernel GPU trace of decode: the rocprofv3 kernel-trace tool preloaded into the engine only, N greedy chats, QUIT (flushes the trace), restart the server
# usage on the box: NAME=k1 [XENV=...] [N=3] [TOK=512] bash tools/kprof.sh ; then python3 tools/kprof.py logs/kprof-k1
# (rocprofv3 around the launcher breaks: its SIGTERM handler kills server.py before QUIT, and a signal to the engine loses the trace)
cd ~/work/rdna3
NAME=${NAME:-k1}; OUT=$PWD/logs/kprof-$NAME; rm -rf "$OUT"; mkdir -p "$OUT/trace"
L=/opt/rocm/lib
TOOL="ROCPROFILER_LIBRARY_CTOR=1 LD_PRELOAD=$L/rocprofiler-sdk/librocprofiler-sdk-tool.so:$L/librocprofiler-sdk.so ROCP_TOOL_LIBRARIES=$L/rocprofiler-sdk/librocprofiler-sdk-tool.so ROCPROF_OUTPUT_FILE_NAME=run_%pid% ROCPROF_OUTPUT_PATH=$OUT/trace ROCPROF_OUTPUT_FORMAT=csv ROCPROF_KERNEL_TRACE=1"
XENV="$TOOL ${XENV:-}" ./strata-rdna3.sh serve > "$OUT/serve.out" 2>&1
python3 - "$OUT" "${N:-3}" "${TOK:-512}" <<'EOF'
import json, sys, time, urllib.request
out, n, tok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ask = "Write a Python module that implements a small LRU cache class with get, put, delete and a resize method, with docstrings and tests."
def chat():
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", headers={"Content-Type": "application/json"},
        data=json.dumps({"messages": [{"role": "user", "content": ask}], "max_tokens": tok, "temperature": 0}).encode())
    return json.load(urllib.request.urlopen(req, timeout=600))
chat()                                   # warm-up (prefill cache, expert placement) outside the window
win = []
for _ in range(n):
    b0 = time.clock_gettime_ns(time.CLOCK_BOOTTIME); t0 = time.monotonic_ns(); r = chat()
    t1 = time.monotonic_ns(); b1 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    win.append({"t0": t0, "t1": t1, "b0": b0, "b1": b1, "tokens": r["usage"]["completion_tokens"], "s": (t1 - t0) / 1e9})
json.dump(win, open(out + "/windows.json", "w"), indent=1)
print(win)
EOF
bash tools/kprof-stop.sh "$OUT" >> "$OUT/serve.out" 2>&1
./strata-rdna3.sh stop >> "$OUT/serve.out" 2>&1
./strata-rdna3.sh serve >> "$OUT/serve.out" 2>&1
find "$OUT/trace" -name '*kernel_trace.csv' | head -3
