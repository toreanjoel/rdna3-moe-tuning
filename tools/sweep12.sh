#!/usr/bin/env bash
# sweep12.sh — EXPDEC Q2_0 down under STRATA_EXP_AUTO (patches/exp-auto.patch; =5 is the old mode 8 down) end to end: interleaved, 8K, then the server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-5} ./strata-rdna3.sh bench ${CTX:-8192} </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
for i in 1 2; do
  run s12old$i "STRATA_EXP_AUTO=5"
  run s12new$i "STRATA_EXP_AUTO=1"
done
} > logs/sweep12.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep12 done" >> logs/sweep12.txt
