#!/usr/bin/env bash
# sweep2.sh — confirm winners of sweep1 (GEMM table 100202 + adapt-swaps), 5 repeats each
cd ~/work/rdna3
T=$HOME/work/rdna3/engine/tools/hip/gfx1100-hipblaslt-100202.txt
run() { NAME=$1 XENV="$2" XARGS="$3" REPEATS=5 ./strata-rdna3.sh bench </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2] XARGS=[$3]"; grep -A3 "input   prefill" logs/bench-$1.log; }
{
run base5     "" ""
run g_swap48  "STRATA_HIPBLASLT_TUNING=$T" "--adapt-swaps 48"
run g_swap64  "STRATA_HIPBLASLT_TUNING=$T" "--adapt-swaps 64"
} > logs/sweep2.txt 2>&1
