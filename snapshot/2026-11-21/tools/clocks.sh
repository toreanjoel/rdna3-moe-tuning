#!/usr/bin/env bash
# clocks.sh — apply GPU clocks/voltage/power the same way start-strata.sh does,
# so A/B runs can change them without touching the user's script or configs.
# usage: clocks.sh <SCLK|-> <MCLK|-> <VO|-> <WATTS|->    ( - = leave at stock )
set -u
DEV=/sys/class/drm/card1/device
OD=$DEV/pp_od_clk_voltage
H=$DEV/hwmon/hwmon5
S=${1:--} M=${2:--} V=${3:--} W=${4:--}
od() { printf "%s\n" "$PW" | sudo -S -p "" tee "$OD" >/dev/null; }
printf "%s\n" "$PW" | sudo -S -p "" tee "$DEV/power/control" >/dev/null 2>/dev/null || true
od r; od c
[[ "$S" != "-" ]] && od "s 1 $S"
[[ "$M" != "-" ]] && od "m 1 $M"
[[ "$V" != "-" ]] && od "vo $V"
od c
if [[ "$W" != "-" ]]; then
  printf "%s\n" "$PW" | sudo -S -p "" bash -c "echo $(( W * 1000000 )) > $H/power1_cap" 2>/dev/null || true
fi
echo "APPLIED sclk=${S} mclk=${M} vo=${V} watts=${W}"
printf "%s\n" "$PW" | sudo -S -v 2>/dev/null; { echo "OD:"; grep -A2 -E "OD_SCLK|OD_MCLK|OD_VDDGFX_OFFSET" "$OD" | tr "\n" " "; echo;
  echo "power_cap=$(cat $H/power1_cap)"; }
