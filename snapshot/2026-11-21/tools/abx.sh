#!/usr/bin/env bash
# abx.sh — interleaved A/B of GPU clock profiles, so expert-cache warm-up drift
# cannot be mistaken for a clock effect. Cycles alternate configs; each cell
# reports median decode tok/s and mean watts.
# usage: abx.sh -c CYCLES "label|SCLK MCLK VO WATTS" "label|..." ...   (- = stock)
set -u
H=/sys/class/drm/card1/device/hwmon/hwmon5
CYC=2
CFG=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -c) CYC=$2; shift 2 ;;
    *) CFG+=("$1"); shift ;;
  esac
done

cell() {                       # $1 label, $2 "S M V W"
  bash "$(dirname "$0")/clocks2.sh" $2 > /tmp/cell_cl.txt
  sed -n '1p' /tmp/cell_cl.txt
  rm -f /tmp/cell_pow.txt
  for i in $(seq 1 120); do cat $H/power1_average >> /tmp/cell_pow.txt; sleep 1; done &
  local sp=$!
  local out
  out=$(python3 "$(dirname "$0")/decode_probe.py" 2 300 112 "$1" | tail -1)
  kill $sp 2>/dev/null
  local w=$(awk '{s+=$1; n++; if($1>m)m=$1} END{if(n) printf "%d %d", s/n/1000000, m/1000000}' /tmp/cell_pow.txt)
  local wavg=${w% *} wmax=${w#* }
  printf "CELL %-26s %s | meanW=%s maxW=%s\n" "$1" "$out" "$wavg" "$wmax"
}

for c in $(seq 1 "$CYC"); do
  for a in "${CFG[@]}"; do
    cell "${a%%|*}" "${a#*|}"
  done
done
