#!/usr/bin/env bash
# foldbench.sh — ms per verify window of kernel switches on the fixed 8K bench prompt (one fresh engine per run, so every arm takes the same windows when the switch is bitwise), ROUNDS alternated, answers compared with the first arm's
# usage (box, via setsid nohup): ARMS="base|;kv|STRATA_QSA_KV_PAIR=1" [ROUNDS=2 REP=2 LEN=8192 OUT=logs/foldbench.txt] tools/foldbench.sh   (an arm is name|ENV ENV ...)
cd ~/work/rdna3
OUT=${OUT:-logs/foldbench.txt}; ROUNDS=${ROUNDS:-2}; LEN=${LEN:-8192}
run() { NAME=fb_$1 XENV="STRATA_DECODE_TIMING=1 $2" REPEATS=${REP:-2} ./strata-rdna3.sh bench $LEN </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A1 "input   prefill" logs/bench-fb_$1.log | tail -1
  grep -h "ms/window = verify" bench/fb_$1/*/engine.log | tail -${REP:-2} | grep -o "[0-9]* windows, avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]*"; }
{
IFS=';' read -ra A <<< "$ARMS"
for round in $(seq 1 $ROUNDS); do
  first=""
  for arm in "${A[@]}"; do IFS='|' read -r n e <<< "$arm"; run "${n}_$round" "$e"
    if [ -z "$first" ]; then first=${n}_$round; else
      for f in bench/fb_$first/*/n$LEN-r*.json; do g=bench/fb_${n}_$round/${f#bench/fb_$first/}
        python3 -c 'import json,sys; a,b=[json.load(open(p))["content"] for p in sys.argv[1:3]]; print(("   SAME " if a==b else "   DIFF ")+sys.argv[2])' "$f" "$g"; done
    fi
  done
done
} >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "foldbench done" >> $OUT
