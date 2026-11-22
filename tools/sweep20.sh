#!/usr/bin/env bash
# sweep20.sh — engine switches off on gfx1100 by default: STRATA_MEGAKERNEL=1 (persistent per-layer GDN kernel, bitwise the six nodes), STRATA_SIDE_MIXER=1/2 (mixer work on graph branches); interleaved, REP 3, 2 rounds, 8K + 32K, server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A2 "input   prefill" logs/bench-$1.log | tail -2 || tail -3 logs/bench-$1.log; }
for i in 1 2; do
  run s20base$i ""
  run s20mk$i "STRATA_MEGAKERNEL=1"
  run s20side1_$i "STRATA_SIDE_MIXER=1"
  run s20side2_$i "STRATA_SIDE_MIXER=2"
done >> logs/sweep20.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep20 done" >> logs/sweep20.txt
