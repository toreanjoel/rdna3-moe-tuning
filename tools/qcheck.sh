#!/usr/bin/env bash
# qcheck.sh — quality check of engine switches: teacher-forced top-1 agreement / KL / perplexity (the fork's quant_quality.py, 5,000 tokens) of each arm against the first arm; the server is stopped for it and restarted at the end
# usage (box, via setsid nohup): ARMS="base|STRATA_X=0;new|STRATA_X=1" [NAME=qc1] tools/qcheck.sh   -> logs/<NAME>.txt   (an arm is name|ENV ENV ...; the default kernel switches and FAST are added to every arm)
cd ~/work/rdna3
NAME=${NAME:-qcheck}; E=$PWD/engine; CFG=$E/rdna3/state/iq3-1gpu-8080.json; Q=$PWD/bench/$NAME; mkdir -p "$Q"
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
run() { local n=$1; shift; local ev=(--env STRATA_RDNA_PAIR=1 --env STRATA_GR_TAIL=1 --env STRATA_EXP_AUTO=1 --env STRATA_ROUTE_TAIL_SKIP=7); for kv in "$@"; do ev+=(--env "$kv"); done
  (cd "$E" && .venv/bin/python tools/quant_quality.py run --config "$CFG" --out "$Q/$n.tsv" --gpu-experts auto "${ev[@]}") > "$Q/$n.log" 2>&1 \
    || echo "$n FAILED (see $Q/$n.log)"; }
{
IFS=';' read -ra A <<< "$ARMS"; first=""
for arm in "${A[@]}"; do IFS='|' read -r n e <<< "$arm"; run "$n" $e
  if [ -z "$first" ]; then first=$n; else echo "== $n [$e] vs $first"; (cd "$E" && .venv/bin/python tools/quant_quality.py compare "$Q/$first.tsv" "$Q/$n.tsv"); fi
done
} > logs/$NAME.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "qcheck done" >> logs/$NAME.txt
