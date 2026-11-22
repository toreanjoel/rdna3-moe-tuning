#!/usr/bin/env bash
# sweep11.sh — per-format expert layouts (STRATA_EXP_AUTO, patches/exp-auto.patch) end to end: interleaved, 8K, then the server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
for i in 1 2; do
  run s11base$i ""
  run s11auto$i "STRATA_EXP_AUTO=1"
  run s11down$i "STRATA_EXP_AUTO=4"
  run s11gu$i "STRATA_EXP_AUTO=3"
done
} > logs/sweep11.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep11 done" >> logs/sweep11.txt
