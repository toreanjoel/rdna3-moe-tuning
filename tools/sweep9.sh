#!/usr/bin/env bash
# sweep9.sh — STRATA_GR_TAIL (patches/gr-tail.patch) end to end: interleaved A/B pairs at 8K, then the server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
for i in 1 2 3; do
  run s9base$i ""
  run s9tail$i "STRATA_GR_TAIL=1"
done
} > logs/sweep9.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep9 done" >> logs/sweep9.txt
