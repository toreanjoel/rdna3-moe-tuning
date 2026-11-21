#!/usr/bin/env bash
# sweep1.sh — A/B of no-output-change engine levers (2026-11-21 research), one bench per arm
cd ~/work/rdna3
T=$HOME/work/rdna3/engine/tools/hip/gfx1100-hipblaslt-100202.txt
run() { NAME=$1 XENV="$2" XARGS="$3" ./strata-rdna3.sh bench </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2] XARGS=[$3]"; grep -A3 "input   prefill" logs/bench-$1.log; }
{
run base   "" ""
run gemm   "STRATA_HIPBLASLT_TUNING=$T" ""
run aux    "STRATA_AUX_CPUS=auto" ""
run pcie15 "" "--pcie-frac 0.15"
run swap48 "" "--adapt-swaps 48"
run pool14 "" "--pool-workers 14"
} > logs/sweep1.txt 2>&1
