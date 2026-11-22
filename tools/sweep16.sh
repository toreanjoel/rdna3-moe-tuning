#!/usr/bin/env bash
# sweep16.sh — IQ4_XS GEMV unroll 2 (patches/kq-unroll.patch; STRATA_RDNA_KQ_UNR=0 = stock) end to end: interleaved, REP 5, 2 rounds, 8K + 32K, then the server back on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-5} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log | tail -2; }
{
for i in 1 2; do
  run s16old$i "STRATA_RDNA_KQ_UNR=0"
  run s16new$i ""
done
} > logs/sweep16.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep16 done" >> logs/sweep16.txt
