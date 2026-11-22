#!/usr/bin/env bash
# sweep22.sh — is the CPU expert path (system RAM bandwidth) on the decode critical path? --pool-workers 6/12/16 (default 12), verify ms/window + tok/s, 8K, REP 2, server back on
cd ~/work/rdna3
OUT=${OUT:-logs/sweep22.txt}
run() { NAME=$1 XENV="STRATA_DECODE_TIMING=1" XARGS="$2" REPEATS=${REP:-2} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XARGS=[$2]"
  grep -A1 "input   prefill" logs/bench-$1.log | tail -1
  grep -h "ms/window = verify" bench/$1/*/engine.log | tail -2 | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]* (GPU-reach wait [0-9.]* + per-layer host [0-9.]* \[[^]]*\]"; }
for w in ${WORKERS:-12 6 16}; do run s22w$w "--pool-workers $w"; done >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep22 done" >> $OUT
