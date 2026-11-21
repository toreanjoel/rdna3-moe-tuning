#!/usr/bin/env bash
# sweep6.sh — confirm the adaptive-lag gain with interleaved A/B pairs, then one decode-timing run (where a step's time goes); restarts the server after
cd ~/work/rdna3
LAG="STRATA_ADAPT_LAG=2 STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_OVERLAP=0"
run() { NAME=$1 XENV="$2" XARGS="$3" REPEATS=${REP:-5} ./strata-rdna3.sh bench </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"; grep -A3 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
run s6baseA "" ""
run s6lagA  "$LAG" ""
run s6baseB "" ""
run s6lagB  "$LAG" ""
REP=1 run s6timing "STRATA_DECODE_TIMING=1" ""
echo "== decode timing lines"; grep -h "decode timing" bench/s6timing/engine.log logs/*.log 2>/dev/null | tail -6
} > logs/sweep6.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep6 done" >> logs/sweep6.txt
