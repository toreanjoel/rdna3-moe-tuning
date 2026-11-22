#!/usr/bin/env bash
# sweep14.sh — confirm sweep13's drafting keepers (--spec-min-p 0.35, --mtp-q4 proj) against the default: interleaved, REP 5, 2 rounds, 8K and 32K, then the server back on
cd ~/work/rdna3
run() {
  NAME=$1 XENV="" XARGS="$2" REPEATS=${REP:-5} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1
  local acc; acc=$(grep -o "drafts accepted [0-9]* of [0-9]*" bench/$1/iq3-1gpu/engine.log | tail -n +2 | awk '{a+=$3; p+=$5} END {printf "accepted %d of %d (%.2f)", a, p, a/p}')
  echo "== $1  XARGS=[$2]  $acc"; grep -A2 "input   prefill" logs/bench-$1.log | tail -2
}
{
for i in 1 2; do
  run s14base$i ""
  run s14p35$i "--spec-min-p 0.35"
  run s14q4p$i "--mtp-q4 proj"
done
} > logs/sweep14.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep14 done" >> logs/sweep14.txt
