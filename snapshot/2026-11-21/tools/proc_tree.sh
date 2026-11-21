#!/usr/bin/env bash
# proc_tree.sh — show the engine's RLIMIT_MEMLOCK and the process chain above it,
# so it is clear which long-lived supervisor needs restarting for a limits.d
# change to reach the engine. usage: proc_tree.sh
P=$(pgrep -f "components/engine.*--serve" | head -1)
echo "engine pid=$P"
grep -i memlock "/proc/$P/limits"
q=$P
for i in 1 2 3 4 5 6; do
  q=$(awk '{print $4}' "/proc/$q/stat" 2>/dev/null)
  [ -z "$q" ] || [ "$q" = "0" ] && break
  cmd=$(tr '\0' ' ' < "/proc/$q/cmdline" 2>/dev/null | cut -c1-90)
  st=$(ps -o lstart= -p "$q" 2>/dev/null | tr -s ' ')
  printf "  parent %s: %s | started %s | memlock %s\n" "$q" "$cmd" "$st" "$(awk '/Max locked memory/{print $4}' "/proc/$q/limits" 2>/dev/null)"
done
