#!/usr/bin/env bash
# sweep4.sh — CPU expert-kernel knobs (exact) and expert-routing switches (change answers), 5 repeats each; restarts the server after
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" XARGS="$3" REPEATS=5 ./strata-rdna3.sh bench </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2] XARGS=[$3]"; grep -A3 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
# exact (same answers)
run s4base    "" ""
run pf4096    "STRATA_IQ_PREFETCH=4096" ""
run pf1024    "STRATA_IQ_PREFETCH=1024" ""
run ymm       "STRATA_CPU_YMM=1" ""
# answer-changing
run tail8     "STRATA_ROUTE_TAIL_SKIP=8" ""
run tail7     "STRATA_ROUTE_TAIL_SKIP=7" ""
run res05     "STRATA_ROUTE_RESIDENT=0.5" ""
run res10     "STRATA_ROUTE_RESIDENT=1.0" ""
run prior05   "STRATA_ROUTE_PRIOR=0.5" ""
run t7res05   "STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_RESIDENT=0.5" ""
} > logs/sweep4.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep4 done" >> logs/sweep4.txt
