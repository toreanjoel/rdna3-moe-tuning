#!/usr/bin/env bash
# sweep19.sh — perf-mode test (NOT a default): SCLK/MCLK above the daily 2300/1300 at the 294 W cap; bench 8K+32K per point with GPU clock/power/temp sampled; daily clocks + server restored at the end
# needs tools/strata-oc installed (passwordless). POINTS="sclk:mclk ..." overrides the grid; REP (default 3).
cd ~/work/rdna3
dev=$(for c in /sys/class/drm/card*/device; do [ "$(cat $c/device 2>/dev/null)" = 0x744c ] && echo $c; done | head -1)
hw=$(ls -d $dev/hwmon/hwmon* | head -1)
# 2300:1500 hard-crashed the box on apply (2026-11-22); 1400 was stable
POINTS=${POINTS:-"2300:1300 2300:1400 2300:1450 2700:1400 3000:1400"}
run() { local s=${1%:*} m=${1#*:} n=s19_${1/:/_}
  echo "-- trying sclk $s mclk $m $(date +%T) (no result after this line = the box crashed here)"; sync
  ( while sleep 1; do echo "$(cat $hw/freq1_input 2>/dev/null) $(cat $hw/freq2_input 2>/dev/null) $(cat $hw/power1_average $hw/power1_input 2>/dev/null | head -1) $(cat $hw/temp2_input $hw/temp3_input 2>/dev/null | tr '\n' ' ')"; done > logs/$n.hw ) & local smp=$!
  SCLK=$s MCLK=$m NAME=$n REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192,32768 </dev/null > logs/$n.out 2>&1; local rc=$?
  kill $smp 2>/dev/null
  echo "== sclk $s mclk $m rc=$rc  $(grep -m1 -E 'applying|already' logs/$n.out)"
  grep -A2 "input   prefill" logs/bench-$n.log | tail -2
  # busy samples only (power > 150 W): mean sclk MHz, mclk MHz, W, hotspot C, mem C
  awk '$3>150e6 {n++; s+=$1; m+=$2; p+=$3; h+=$4; t+=$5} END {if(n) printf "   busy n=%d sclk %.0f mclk %.0f  %.0f W  hotspot %.0f C  mem %.0f C\n", n, s/n/1e6, m/n/1e6, p/n/1e6, h/n/1e3, t/n/1e3}' logs/$n.hw
}
for p in $POINTS; do run $p >> logs/sweep19.txt 2>&1; done
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1   # back to the daily SCLK 2300 / MCLK 1300
echo "sweep19 done" >> logs/sweep19.txt
