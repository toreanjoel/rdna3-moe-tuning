#!/usr/bin/env bash
# power_guard.sh — keep the 7900 XTX under a watt budget without permanently
# crippling decode: the card refuses power1_cap below 294 W, so the only way to
# hold ~200 W is to cap the core clock. Decode sits at ~176 W on its own, but a
# prompt read spikes to 220-260 W. This guard runs the clock high and drops it
# only while the card is actually over budget, then restores it.
#
# usage: PW=<sudo password> power_guard.sh [HIGH_SCLK] [LOW_SCLK] [BUDGET_W] [MCLK] [VO]
#        defaults:      2300      1500       200      1300   -50
# stop:  kill it; it leaves the last applied clocks in place (put your own back
#        with clocks2.sh).
set -u
DEV=/sys/class/drm/card1/device
OD=$DEV/pp_od_clk_voltage
H=$DEV/hwmon/hwmon5
HIGH=${1:-2300} LOW=${2:-1500} BUDGET=${3:-200} M=${4:-1300} V=${5:--50}
ON=$(( BUDGET - 8 ))          # drop clocks above this
OFF=$(( BUDGET - 35 ))        # restore once comfortably below
STATE=""

setclk() {
  printf "%s\n" "$PW" | sudo -S -p "" bash -c \
    "echo 's 1 $1' > $OD; echo 'm 1 $M' > $OD; echo 'vo $V' > $OD; echo c > $OD" 2>/dev/null
  STATE=$1
  echo "$(date +%H:%M:%S) sclk->$1"
}

printf "%s\n" "$PW" | sudo -S -v 2>/dev/null
( while :; do sleep 240; printf "%s\n" "$PW" | sudo -S -v 2>/dev/null; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT

setclk "$HIGH"
while :; do
  W=$(( $(cat $H/power1_average) / 1000000 ))
  if [[ "$W" -gt "$ON" && "$STATE" != "$LOW" ]]; then
    setclk "$LOW"
  elif [[ "$W" -lt "$OFF" && "$STATE" != "$HIGH" ]]; then
    sleep 3                                   # let the load settle before speeding up
    W2=$(( $(cat $H/power1_average) / 1000000 ))
    [[ "$W2" -lt "$OFF" ]] && setclk "$HIGH"
  fi
  sleep 0.3
done
