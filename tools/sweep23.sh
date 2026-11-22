#!/usr/bin/env bash
# sweep23.sh — generic env A/B (SETS="envA;envB;..", P=name prefix, OUT=log; first used for STRATA_PREFETCH), 8K, REP 2, decode tok/s + verify ms, server back on
cd ~/work/rdna3
OUT=${OUT:-logs/sweep23.txt}
run() { NAME=$1 XENV="STRATA_DECODE_TIMING=1 $2" REPEATS=${REP:-2} ./strata-rdna3.sh bench ${CTXS:-8192} </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A2 "input   prefill" logs/bench-$1.log | tail -2
  grep -h "ms/window = verify" bench/$1/*/engine.log | tail -2 | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]*"; }
IFS=';' read -ra SETS <<< "${SETS:-;STRATA_PREFETCH=1;STRATA_PREFETCH=1 STRATA_PREFETCH_GRID=48;STRATA_PREFETCH=3;STRATA_PREFETCH=2}"
P=${P:-s23_}; i=0; for e in "${SETS[@]}"; do run $P$i "$e"; i=$((i+1)); done >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep23 done" >> $OUT
