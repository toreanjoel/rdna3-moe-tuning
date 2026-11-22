#!/usr/bin/env bash
# sweep26.sh — STRATA_GU_SWIGLU=1 (expert gate/up + SwiGLU + q8_1 in one launch, off on HIP by default) vs off, alternated, 8K, REP 2; answers diffed for bitwise
cd ~/work/rdna3
OUT=${OUT:-logs/sweep26.txt}
run() { NAME=$1 XENV="STRATA_DECODE_TIMING=1 $2" REPEATS=${REP:-2} ./strata-rdna3.sh bench 8192 </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2]"
  grep -A1 "input   prefill" logs/bench-$1.log | tail -1
  grep -h "all .* requests passed\|FAIL" logs/bench-$1.log | tail -1
  grep -h "ms/window = verify" bench/$1/*/engine.log | tail -2 | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]*"; }
{
run s26_0 ""
run s26_1 "STRATA_GU_SWIGLU=1"
run s26_2 ""
run s26_3 "STRATA_GU_SWIGLU=1"
for p in 0 2; do q=$((p+1)); for f in bench/s26_$p/*/n8192-r*.json; do
  g=bench/s26_$q/${f#bench/s26_$p/}
  python3 - "$f" "$g" <<'EOF'
import json,sys
def txt(p): return json.load(open(p))["content"]
a,b=txt(sys.argv[1]),txt(sys.argv[2]); print(("SAME " if a==b else "DIFF ")+sys.argv[2])
EOF
done; done
} >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep26 done" >> $OUT
