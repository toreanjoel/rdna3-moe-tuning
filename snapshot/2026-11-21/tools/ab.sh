#!/usr/bin/env bash
# ab.sh — one A/B cell: apply clocks, probe decode, report tok/s with mean watts/clocks.
# usage: ab.sh <SCLK> <MCLK> <VO> <WATTS> <label>   ( - = stock/leave )
set -u
DEV=/sys/class/drm/card1/device
H=$DEV/hwmon/hwmon5
S=${1:--} M=${2:--} V=${3:--} W=${4:--} L=${5:-cell}
bash "$(dirname "$0")/clocks2.sh" "$S" "$M" "$V" "$W" >/dev/null
rm -f /tmp/ab_pow.txt
for i in $(seq 1 90); do
  printf "%s %s %s\n" "$(cat $H/power1_average)" "$(cat $H/freq1_input)" "$(cat $H/freq2_input)" >> /tmp/ab_pow.txt
  sleep 1
done &
SP=$!
OUT=$(python3 "$(dirname "$0")/decode_probe.py" 4 300 112 "$L" | tail -1)
kill $SP 2>/dev/null; wait 2>/dev/null
read PW SC MC <<< "$(awk '{wp+=$1; ws+=$2; wm+=$3; n++} END{if(n)printf "%d %d %d", wp/n, ws/n, wm/n}' /tmp/ab_pow.txt)"
printf "CELL %-30s %s | mean %.0f W | sclk %.0f mclk %.0f\n" "$L" "$OUT" "$(echo "$PW/1000000" | bc)" "$(echo "$SC/1000000" | bc)" "$(echo "$MC/1000000" | bc)"
printf "cap now %s min %s\n" "$(cat $H/power1_cap)" "$(cat $H/power1_cap_min)"
