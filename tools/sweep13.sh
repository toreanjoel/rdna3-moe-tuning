#!/usr/bin/env bash
# sweep13.sh — drafting knobs end to end at 8K (MTP --spec / --spec-min-p / --mtp-q4; DFlash2 via SET=df): tok/s and drafts accepted per run, then the server back on
cd ~/work/rdna3
run() {
  NAME=$1 XENV="" XARGS="$2" REPEATS=${REP:-3} ./strata-rdna3.sh bench ${CTX:-8192} </dev/null >/dev/null 2>&1
  local e=bench/$1/iq3-1gpu/engine.log
  local acc; acc=$(grep -o "generated in [0-9]* ms ([0-9.]* tok/s), drafts accepted [0-9]* of [0-9]*" "$e" | tail -n +2 |
    awk '{ms+=$3; a+=$9; p+=$11; n++} END {if (n) printf "runs %d  accepted %d of %d (%.2f)  windows/256 %.0f  ms/window %.1f", n, a, p, a/p, 256-a/n, ms/n/(256-a/n)}')
  echo "== $1  XARGS=[$2]  $acc"; grep -A1 "input   prefill" logs/bench-$1.log | tail -1
}
DF="--draft dflash2 --draft-model $PWD/data/dflash2/qwen3.8-flash-next-dflash2.q8.pack"
{
if [ "${SET:-mtp}" = df ]; then
run s13mq "--spec-min-p 0.35 --mtp-q4 proj"
run s13df "$DF"
run s13dfc6 "$DF --spec 6 --draft-cap 6"
run s13dfc8 "$DF --spec 8 --draft-cap 8"
run s13dfp3 "$DF --spec 6 --draft-cap 6 --draft-conf 0.3"
else
run s13base ""
run s13spec3 "--spec 3"
run s13spec5 "--spec 5"
run s13spec6 "--spec 6"
run s13minp35 "--spec-min-p 0.35"
run s13minp70 "--spec-min-p 0.7"
run s13s6p35 "--spec 6 --spec-min-p 0.35"
run s13q4proj "--mtp-q4 proj"
fi
} > logs/sweep13${SET:+-$SET}.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep13 done" >> logs/sweep13${SET:+-$SET}.txt
