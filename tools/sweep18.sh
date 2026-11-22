#!/usr/bin/env bash
# sweep18.sh — Q4_K branchless scales (STRATA_RDNA_KQ_SCB) + RDNA Q4_K from 1 column (STRATA_RDNA_KQ_MIN=2 default) end to end: base = KQ_MIN=1 SCB=0 (d58d76f), scb = KQ_MIN=1 (must match base's answers), new = defaults; interleaved, REP 5, 2 rounds, 8K + 32K, then the server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-5} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log | tail -2; }
{
for i in 1 2; do
  run s18base$i "STRATA_RDNA_KQ_MIN=1 STRATA_RDNA_KQ_SCB=0"
  run s18scb$i "STRATA_RDNA_KQ_MIN=1"
  run s18new$i ""
done
} > logs/sweep18.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep18 done" >> logs/sweep18.txt
