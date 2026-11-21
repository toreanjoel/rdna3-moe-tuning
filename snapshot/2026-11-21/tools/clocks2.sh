#!/usr/bin/env bash
# clocks2.sh — apply GPU clocks/voltage/power in ONE sudo shell (separate sudo
# tee calls lose OD writes on this box), then read back the OD table so a failed
# write is visible. Needs $PW for sudo -S. Mirrors start-strata.sh's method.
# usage: clocks2.sh <SCLK|-> <MCLK|-> <VO|-> <WATTS|->    ( - = leave at stock )
set -u
DEV=/sys/class/drm/card1/device
OD=$DEV/pp_od_clk_voltage
H=$DEV/hwmon/hwmon5
S=${1:--} M=${2:--} V=${3:--} W=${4:--}

CMD="echo on > $DEV/power/control 2>/dev/null; echo r > $OD; echo c > $OD;"
[[ "$S" != "-" ]] && CMD="$CMD echo 's 1 $S' > $OD;"
[[ "$M" != "-" ]] && CMD="$CMD echo 'm 1 $M' > $OD;"
[[ "$V" != "-" && "$V" != "0" ]] && CMD="$CMD echo 'vo $V' > $OD;"
[[ "$W" != "-" ]] && CMD="$CMD echo $(( W * 1000000 )) > $H/power1_cap;"
CMD="$CMD echo c > $OD;"

printf "%s\n" "$PW" | sudo -S -p "" bash -c "$CMD" 2>/dev/null
sleep 1
RB=$(printf "%s\n" "$PW" | sudo -S -p "" sh -c "grep -A1 -E 'OD_SCLK|OD_MCLK|OD_VDDGFX' $OD" 2>/dev/null | tr '\n' ' ')
printf "APPLIED sclk=%s mclk=%s vo=%s watts=%s | readback: %s | cap_W=%s\n" \
  "$S" "$M" "$V" "$W" "$RB" "$(( $(cat $H/power1_cap) / 1000000 ))"
