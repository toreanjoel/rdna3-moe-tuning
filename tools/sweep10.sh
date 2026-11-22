#!/usr/bin/env bash
# sweep10.sh — merge launches with existing bitwise switches: STRATA_RDNA_PAIR (shared-expert gate+up), STRATA_LFUSE: interleaved A/B pairs at 8K, then the server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
for i in 1 2; do
  run s10base$i ""
  run s10pair$i "STRATA_RDNA_PAIR=1"
  run s10lfuse$i "STRATA_LFUSE=1"
  run s10both$i "STRATA_RDNA_PAIR=1 STRATA_GR_TAIL=1"
done
} > logs/sweep10.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep10 done" >> logs/sweep10.txt
