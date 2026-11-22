#!/usr/bin/env bash
# sweep21.sh — timing ablations (STRATA_ABL bitmask, wrong output, ms/window only): 1 = skip shared expert, 2 = skip VRAM expert GEMVs, 4 = skip mixer dense GEMVs; fixed window (FIX, default --spec-min-p 0) so T stays put; 8K, server back on
cd ~/work/rdna3
OUT=${OUT:-logs/sweep21.txt}
run() { NAME=$1 XENV="STRATA_DECODE_TIMING=1 $2" XARGS="${FIX:---spec-min-p 0}" REPEATS=${REP:-1} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A1 "input   prefill" logs/bench-$1.log | tail -1
  grep -h "ms/window = verify" bench/$1/*/engine.log | tail -2 | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]* (GPU-reach wait [0-9.]* + per-layer host [0-9.]*"; }
for m in ${MASKS:-0 1}; do run s21f$m "STRATA_ABL=$m"; done >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep21 done" >> $OUT
