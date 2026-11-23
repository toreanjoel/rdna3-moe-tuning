#!/usr/bin/env bash
# sweep46.sh — --spec 5 with the retrained head (MTP=ft) on the 8K/32K bench: won varied chats +0.3% (sweep45); check long context doesn't lose
cd ~/work/rdna3
run() { NAME=$1 XARGS="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1; echo "== $1  XARGS=[$2]"; grep -A3 "input   prefill" logs/bench-$1.log || tail -4 logs/bench-$1.log; }
{
for i in 1 2; do
  run s46base$i ""
  run s46spec5$i "--spec 5"
done
} > logs/sweep46.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep46 done" >> logs/sweep46.txt
