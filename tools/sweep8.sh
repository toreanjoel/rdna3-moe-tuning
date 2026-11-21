#!/usr/bin/env bash
# sweep8.sh — HIP runtime knobs that change per-node dispatch cost (a node costs ~3.4 us, sweep7): kernel args in VRAM, graph packet capture; A/B/A; restarts the server after
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A2 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
run s8base0 ""
run s8karg  "HIP_FORCE_DEV_KERNARG=1"
run s8nocap "DEBUG_CLR_GRAPH_PACKET_CAPTURE=0"
run s8base1 ""
run s8karg1 "HIP_FORCE_DEV_KERNARG=1"
} > logs/sweep8.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep8 done" >> logs/sweep8.txt
