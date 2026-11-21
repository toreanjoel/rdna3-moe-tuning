#!/usr/bin/env bash
# sweep7.sh — what one graph node costs in a decode window: STRATA_DUMMY_NODES=k adds k empty kernels per layer; A/B/A pattern; restarts the server after
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
run s7base0 ""
run s7d8    "STRATA_DUMMY_NODES=8"
run s7d24   "STRATA_DUMMY_NODES=24"
run s7base1 ""
} > logs/sweep7.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep7 done" >> logs/sweep7.txt
