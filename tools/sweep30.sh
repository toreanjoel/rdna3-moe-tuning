#!/usr/bin/env bash
# sweep30.sh — untested MTP draft switches on gfx1100: STRATA_MTP_QFUSE=1, STRATA_MTP_MULTICOPY=1, STRATA_MTP_HIP_PRIO=1, all three; 8K, REP 2, answers diffed vs base
cd ~/work/rdna3
OUT=${OUT:-logs/sweep30.txt}
run() { NAME=$1 XENV="STRATA_DECODE_TIMING=1 $2" REPEATS=${REP:-2} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A1 "input   prefill" logs/bench-$1.log | tail -1
  grep -h "ms/window = verify" bench/$1/*/engine.log | tail -2 | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]*.*draft [0-9.]*" | sed -E "s/\(GPU-reach.*\) \+ commit/... commit/"; }
same() { for f in bench/$1/*/n8192-r*.json; do g=bench/$2/${f#bench/$1/}; python3 -c "import json,sys;a,b=(json.load(open(p))['content'] for p in sys.argv[1:]);print(('SAME ' if a==b else 'DIFF ')+sys.argv[2])" "$f" "$g"; done; }
{
run s30_0 ""
run s30_q "STRATA_MTP_QFUSE=1"
run s30_m "STRATA_MTP_MULTICOPY=1"
run s30_p "STRATA_MTP_HIP_PRIO=1"
run s30_a "STRATA_MTP_QFUSE=1 STRATA_MTP_MULTICOPY=1 STRATA_MTP_HIP_PRIO=1"
run s30_1 ""
for n in q m p a 1; do same s30_0 s30_$n; done
} >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep30 done" >> $OUT
