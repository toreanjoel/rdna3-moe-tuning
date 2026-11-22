#!/usr/bin/env bash
# kprof-stop.sh — end a profiled server gently (SIGTERM to server.py only -> QUIT to the engine), report how long the engine took to exit and what rocprofv3 wrote
# usage on the box: bash tools/kprof-stop.sh logs/kprof-<name>
cd ~/work/rdna3
OUT=${1:?trace dir}
E=$(pgrep -f "^$HOME/work/rdna3/engine/engine/strata --serve")
pkill -TERM -f "^$HOME/work/rdna3/engine/.venv/bin/python $HOME/work/rdna3/engine/serve/server" || true
for i in $(seq 120); do sleep 1; [ -n "$E" ] && kill -0 "$E" 2>/dev/null || { echo "engine gone after $i s"; break; }; done
cp logs/serve.out "$OUT/server-stderr.out" 2>/dev/null
ls -la "$OUT/trace" 2>&1 | head -5
grep -v simple_timer logs/serve.out | tail -8 | cut -c1-220
