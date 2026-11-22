#!/usr/bin/env bash
# sweep28.sh — per-stage GPU profile of the current build (STRATA_VERIFY_PROFILE=1: stamp nodes, small stages inflated ~2x), 8K, REP 1, server back on
cd ~/work/rdna3
OUT=${OUT:-logs/sweep28.txt}
NAME=s28p XENV="STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1" REPEATS=1 ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1
{ echo "== s28p"; grep -h "decode timing:\|decode GPU stages" bench/s28p/*/engine.log | tail -4; } >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep28 done" >> $OUT
