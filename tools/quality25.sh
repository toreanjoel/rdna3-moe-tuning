#!/usr/bin/env bash
# quality25.sh — teacher-forced KL / top-1 of STRATA_RDNA_MIX=1 (qkv+z and shared gate+up as one mixed-format RDNA launch) vs the current default (=0), via quant_quality.py
cd ~/work/rdna3
E=$PWD/engine; CFG=$E/rdna3/state/iq3-1gpu-8080.json; Q=$PWD/bench/quality25; mkdir -p "$Q"
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
run() { local n=$1; shift; local ev=(--env STRATA_RDNA_PAIR=1 --env STRATA_GR_TAIL=1 --env STRATA_EXP_AUTO=1 --env STRATA_ROUTE_TAIL_SKIP=7); for kv in "$@"; do ev+=(--env "$kv"); done
  (cd "$E" && .venv/bin/python tools/quant_quality.py run --config "$CFG" --out "$Q/$n.tsv" --gpu-experts auto "${ev[@]}") > "$Q/$n.log" 2>&1 \
    || echo "$n FAILED (see $Q/$n.log)"; }
{
run base STRATA_RDNA_MIX=0
run mix1 STRATA_RDNA_MIX=1

cd "$E" && for n in mix1; do
  [ -s "$Q/$n.tsv" ] && { echo "== $n"; .venv/bin/python tools/quant_quality.py compare "$Q/base.tsv" "$Q/$n.tsv"; }; done
} > "$PWD/logs/quality25.txt" 2>&1
cd ~/work/rdna3 && ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "quality25 done" >> logs/quality25.txt
