#!/usr/bin/env bash
# quality4.sh — teacher-forced KL / top-1 of the routing switches against the current default (real VRAM expert cache), via the engine's quant_quality.py
cd ~/work/rdna3
[ "${1:-}" = "--after-sweep4" ] && until grep -q "sweep4 done" logs/sweep4.txt 2>/dev/null; do sleep 30; done
E=$PWD/engine; CFG=$E/rdna3/state/iq3-1gpu-8080.json; Q=$PWD/bench/quality4; mkdir -p "$Q"
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
run() { local n=$1; shift; local ev=(); for kv in "$@"; do ev+=(--env "$kv"); done
  (cd "$E" && .venv/bin/python tools/quant_quality.py run --config "$CFG" --out "$Q/$n.tsv" --gpu-experts auto "${ev[@]}") > "$Q/$n.log" 2>&1 \
    || echo "$n FAILED (see $Q/$n.log)"; }
{
run ref
run ref2
run tail8   STRATA_ROUTE_TAIL_SKIP=8
run tail7   STRATA_ROUTE_TAIL_SKIP=7
run res05   STRATA_ROUTE_RESIDENT=0.5
run res10   STRATA_ROUTE_RESIDENT=1.0
run prior05 STRATA_ROUTE_PRIOR=0.5
run t7res05 STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_RESIDENT=0.5
cd "$E" && for n in ref2 tail8 tail7 res05 res10 prior05 t7res05; do
  [ -s "$Q/$n.tsv" ] && { echo "== $n"; .venv/bin/python tools/quant_quality.py compare "$Q/ref.tsv" "$Q/$n.tsv"; }; done
} > logs/quality4.txt 2>&1
cd ~/work/rdna3 && ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "quality4 done" >> logs/quality4.txt
