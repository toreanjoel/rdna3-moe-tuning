#!/usr/bin/env bash
# sweep32.sh — STRATA_WAIT_PF (combine's CPU-expert wait warms layer l+1's weights in the MALL): off / 96 / 48 blocks / 96 at 8 uint4; 8K+32K, REP 3, interleaved, answers diffed vs off
cd ~/work/rdna3
OUT=${OUT:-logs/sweep32.txt}
run() { NAME=$1 XENV="STRATA_DECODE_TIMING=1 $2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A2 "input   prefill" logs/bench-$1.log | tail -2
  grep -h "ms/window = verify" bench/$1/*/engine.log | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]* (GPU-reach wait [0-9.]*" | tail -2; }
same() { for f in bench/$1/*/n*-r*.json; do g=bench/$2/${f#bench/$1/}; python3 -c "import json,sys;a,b=(json.load(open(p))['content'] for p in sys.argv[1:]);print(('SAME ' if a==b else 'DIFF ')+sys.argv[2])" "$f" "$g"; done | sort | uniq -c; }
{
for round in 1 2; do
run s32_0_$round ""
run s32_96_$round "STRATA_WAIT_PF=96"
run s32_48_$round "STRATA_WAIT_PF=48"
run s32_96c8_$round "STRATA_WAIT_PF=96 STRATA_WAIT_PF_CH=8"
done
for n in 96 48 96c8; do for r in 1 2; do echo "-- s32_${n}_$r vs s32_0_1"; same s32_0_1 s32_${n}_$r; done; done
} >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep32 done" >> $OUT
