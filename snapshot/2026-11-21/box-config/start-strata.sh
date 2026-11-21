#!/usr/bin/env bash
# start-strata.sh - apply the RX 7900 XTX and CPU settings, then (re)start Strata.
# Usage:  ./start-strata.sh              apply settings and start (baseline)
#         ./start-strata.sh a            + experiment A: faster generation kernels
#         ./start-strata.sh b            + experiment B: faster prompt-reading kernels
#         ./start-strata.sh a b          + both
#         ./start-strata.sh --plan [a b] only show what Strata would do (changes nothing)
set -euo pipefail

# PLE table needs unlimited memlock; a terminal opened before
# /etc/security/limits.d/60-strata-memlock.conf still carries the old 12063432 KB cap, so the
# engine would load for ~17 min with the PLE table unlocked. Raise first, then check the result.
ulimit -l unlimited 2>/dev/null || true
if [[ "$(ulimit -l)" != "unlimited" ]]; then
  echo "!! memlock still $(ulimit -l) KB - open a NEW terminal (limits.d only applies to new logins) and run this again."
  echo "   to load capped anyway: MEMLOCK_OK=1 ./start-strata.sh"
  [[ ${MEMLOCK_OK:-} == 1 ]] || exit 1
fi

ulimit -l unlimited 2>/dev/null || true


# ---- settings -------------------------------------------------------------
# Each line shows:  value now   # what it does | stock = card/Strata default | maxed = your fastest measured run
#
# MAXED run (Nov 20): ~2,000 tok/s prompt reading at ~350 W, 71-76 tok/s generation at 75K context.
#   To go back to it: POWER_CAP=402  MAX_SCLK=""  VOLT_OFFSET=0  (everything else as below)
# NOW: lower-power version of it, aiming for ~1,500 tok/s prompt reading at roughly 200 W.
#   Note: before the commit fix (Nov 20), MAX_SCLK and MAX_MCLK were shown but never applied.

GPU=0                  # amd-smi index of the 7900 XTX

# -- GPU power and clocks --
POWER_CAP=294          # watts, card allows 294-402        | maxed: 402
                       #   with a clock cap the card stays well under this; it only matters at stock clocks
MAX_SCLK=2300          # max core clock, MHz               | stock: "" (no cap; card runs ~2,550 when reading) | maxed: ""
                       #   lower = slower prompt reading at lower power. Untested since the commit fix: start at 1700,
                       #   check freq1_input and watts during a big read, then move up or down 100 at a time
VOLT_OFFSET=-50        # core undervolt, mV (0 to -450)     | stock: 0 | maxed: 0
                       #   step -25 per run; crash, freeze or driver reset = go back one step; reboot clears it
MAX_MCLK=1300          # max VRAM clock, MHz (up to 1500)   | stock: 1250 | maxed: 1300
                       #   generation waits on VRAM; +25 per run while tok/s rises; if it drops or crashes, back one step

# -- model --
MODEL=iq3_s            # quant                              | maxed: iq3_s (matches the full model on benchmarks)
CONTEXT=128k           # 32k | 64k | 128k                    | maxed: 128k (costs no expert slots; agent compacts at ~85K)
VISION=off             # image input                        | maxed: off (its VRAM goes to experts)

# -- drafting (guessing tokens ahead) --
DRAFTING=dflash2       # auto | mtp | dflash2 | off          | maxed: dflash2
DRAFT_VOCAB=en         # drafter vocabulary                 | maxed: en (English/code)
SPEC=3                 # draft depth                        | stock: "" (Strata picks; 8 on yours) | maxed: 3
                       #   3 beat the default: 61.2 vs 56.7 tok/s. "" = Strata's default
SPEC_MIN_P=""          # MTP draft cut-off; leave ""         | stock: "" | maxed: ""
                       #   DFlash2 ignores it: its own cut-off is --draft-conf, which strata launch already sets to 0.5
MTP=off                # built-in MTP draft layer           | maxed: off (its ~0.7-1 GB of VRAM goes to experts)

# -- CPU (computes part of every token's experts) --
CPU_EPP=performance  # performance | balance_performance | balance_power | power   | maxed: performance
CPU_BOOST=1            # 1 = boost on, 0 = off              | stock: 1 | maxed: 1
                       #   the CPU computes part of every token's experts; 0 = quieter fan but slower generation

# -- other --
ACCESS=local           # local = only this PC answers; reach it from the Mac via an SSH tunnel
TUI=on                 # on = Strata dashboard (run inside tmux), off = plain log output
STRATA_CONFIG=""       # path to your live strata-<model>.json; "" = find it automatically
                       #   (skips repo/benchmark copies; set it here to be sure)

# Engine switches passed with `strata launch --set NAME=VALUE`. Remove a line to turn it off.
STRATA_ENV=(
  STRATA_HIPBLASLT_TABLE=on   # gfx1100 prompt-speed table   | stock: off | maxed: on (~600 -> ~2,000 tok/s with stock clocks)
)

# EXPERIMENT A (./start-strata.sh a): generation switches Strata turns on for Strix Halo/RDNA4 but not the 7900 XTX.
EXP_A=(
  STRATA_EXPERT_V2=1 STRATA_TSUM=1        # grouped decode experts for your IQ3_S/IQ4_NL pack; same results
  STRATA_ATTN_LANECELL=1 STRATA_Q8_PACKED=1 STRATA_MMVF_ROWS=1 STRATA_LFUSE=1 STRATA_QFUSE=1
  STRATA_GDN_HEAD=1 STRATA_GDN_PP=2 STRATA_GDN_CONVL2=1 STRATA_GDN_NOY=1 STRATA_GDN_SPLIT=1   # DeltaNet layers
  STRATA_PLE_BATCH=1
)
# EXPERIMENT B (./start-strata.sh b): prompt reading on the matrix cores, on top of the hipBLASLt table.
EXP_B=(
  STRATA_HIP_WMMA=1 STRATA_PA_FAST=1 STRATA_SELECT_WMMA=1   # prompt attention and block scorer
  STRATA_PF_GEMM=1 STRATA_PF_PAD=1 STRATA_HC_UPMIX=1        # prompt projections
  STRATA_PF_FUSED=1                                         # prompt experts, fused int8 kernels
)
# ---------------------------------------------------------------------------

PLAN=0
for arg in "$@"; do
  case "$arg" in
    --plan) PLAN=1 ;;
    a|A)    STRATA_ENV+=("${EXP_A[@]}"); echo "Experiment A: on" ;;
    b|B)    STRATA_ENV+=("${EXP_B[@]}"); echo "Experiment B: on" ;;
    *)      echo "Unknown argument: $arg (use a, b or --plan)"; exit 1 ;;
  esac
done

STRATA_ARGS=(
  --family qwen --model "$MODEL"
  --context "$CONTEXT" --vision "$VISION"
  --drafting "$DRAFTING" --draft-vocab "$DRAFT_VOCAB" --mtp "$MTP"
  --access "$ACCESS"
  --yes
)
if [[ "$TUI" == "off" ]]; then STRATA_ARGS+=(--no-tui); fi
for e in "${STRATA_ENV[@]}"; do STRATA_ARGS+=(--set "$e"); done

# --spec and --spec-min-p are engine flags, not `strata launch` options, so they go in
# strata-<model>.json ("args" and "user_args"; launch passes on "user_args"). The first change saves the original as .orig;
# an empty value puts back whatever the original file had for that flag.
set_engine_args() {
  local cfg="$STRATA_CONFIG"
  if [[ -z "$cfg" ]]; then
    # Only this model's config, never copies inside a source checkout (docs/bench-results, tests, ...),
    # newest first, so the file strata launch last wrote wins.
    mapfile -t found < <(find "$HOME" -maxdepth 6 -type f -name "strata-${MODEL}.json" \
        -not -path '*/docs/*' -not -path '*/bench-results/*' -not -path '*/tests/*' \
        -not -path '*/.git/*' -not -path '*/node_modules/*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | cut -d' ' -f2-)
    if (( ${#found[@]} == 0 )); then
      echo "!! No strata-${MODEL}.json found under $HOME. Draft settings left as is."
      echo "   Find it with: find / -name 'strata-*.json' 2>/dev/null   and set STRATA_CONFIG= at the top."
      return 0
    fi
    cfg="${found[0]}"
    if (( ${#found[@]} > 1 )); then
      echo "Found several configs, using the newest. Set STRATA_CONFIG= to pick another:"
      printf '   %s\n' "${found[@]}"
    fi
  fi
  echo "Config: $cfg"
  python3 - "$cfg" --spec "$SPEC" --spec-min-p "$SPEC_MIN_P" <<'PY' || echo "!! Could not update the config. Draft settings left as is."
import json, os, shutil, sys
path, pairs = sys.argv[1], sys.argv[2:]
orig = path + ".orig"
if not os.path.exists(orig):
    shutil.copy2(path, orig)

def strip(args, flag):
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        if isinstance(args[i], str) and args[i].startswith(flag + "="):
            i += 1
            continue
        out.append(args[i])
        i += 1
    return out

def original(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return [flag, args[i + 1]]
        if isinstance(a, str) and a.startswith(flag + "="):
            return [a]
    return []

with open(path) as f:
    cfg = json.load(f)
with open(orig) as f:
    base = json.load(f)
# strata launch rebuilds the engine command from its own flags plus "user_args",
# so the same change goes into both "args" and "user_args".
for key in ("args", "user_args"):
    if key not in cfg and key == "user_args" and not any(pairs[1::2]):
        continue
    lst = cfg.get(key, [])
    for flag, value in zip(pairs[0::2], pairs[1::2]):
        lst = strip(lst, flag)
        lst += [flag, value] if value else original(base.get(key, []), flag)
    cfg[key] = lst
for flag, value in zip(pairs[0::2], pairs[1::2]):
    print(f"{flag}: {value if value else 'as originally set up'}")
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
}

if [[ "$PLAN" == 1 ]]; then
  strata launch --plan "${STRATA_ARGS[@]}"
  exit 0
fi

echo "== GPU power settings =="
MIN_CAP=$(amd-smi static -g "$GPU" --limit | awk '/PPT0/{f=1} f && /MIN_POWER_LIMIT/{print $2; exit}')
MAX_CAP=$(amd-smi static -g "$GPU" --limit | awk '/PPT0/{f=1} f && /MAX_POWER_LIMIT/{print $2; exit}')
if [[ -n "$MIN_CAP" && "$POWER_CAP" -lt "$MIN_CAP" ]]; then
  echo "Card minimum is ${MIN_CAP} W, so using ${MIN_CAP} W instead of ${POWER_CAP} W."
  POWER_CAP=$MIN_CAP
fi
if [[ "$MAX_CAP" =~ ^[0-9]+$ && "$POWER_CAP" -gt "$MAX_CAP" ]]; then
  echo "Card maximum is ${MAX_CAP} W, so using ${MAX_CAP} W instead of ${POWER_CAP} W."
  POWER_CAP=$MAX_CAP
fi
# amd-smi reports success even when it refuses, so check its message instead
out=$(sudo amd-smi set -g "$GPU" --power-cap ppt0 "$POWER_CAP" 2>&1 || true)
if grep -qiE "must be|not_supported|unable|error" <<<"$out"; then echo "!! Power cap not applied: $out"; fi

# Clocks and voltage are written straight to the driver's pp_od_clk_voltage file and then
# committed with "c". Without the commit the file SHOWS the new values but the card ignores them.
# Every run starts from stock (r), so empty settings really mean stock.
DEV=$(dirname "$(grep -l 0x744c /sys/class/drm/card*/device/device 2>/dev/null | head -1)" 2>/dev/null || true)
OD=""
if [[ -n "$DEV" && -e "$DEV/pp_od_clk_voltage" ]]; then OD="$DEV/pp_od_clk_voltage"; fi
od() { echo "$1" | sudo tee "$OD" >/dev/null; }
if [[ -n "$OD" ]]; then
  echo on | sudo tee "$DEV/power/control" >/dev/null || true   # a sleeping GPU refuses clock changes
  od r && od c && echo "Clocks and voltage reset to stock" || echo "!! Could not reset to stock. Continuing."
  if [[ -n "$MAX_SCLK" ]]; then
    od "s 1 $MAX_SCLK" && echo "Core clock max: ${MAX_SCLK} MHz" \
      || echo "!! Core clock not accepted (outside the card's OD_RANGE?)."
  fi
  if [[ -n "$MAX_MCLK" ]]; then
    od "m 1 $MAX_MCLK" && echo "VRAM clock max: ${MAX_MCLK} MHz" \
      || echo "!! VRAM clock not accepted (outside the card's OD_RANGE?)."
  fi
  if [[ -n "$VOLT_OFFSET" && "$VOLT_OFFSET" != "0" ]]; then
    od "vo $VOLT_OFFSET" && echo "Core voltage offset: ${VOLT_OFFSET} mV" \
      || echo "!! Voltage offset not accepted (outside VDDGFX_OFFSET range?)."
  fi
  od c && echo "Clock and voltage changes committed" || echo "!! Commit failed. Clocks may be at stock."
else
  echo "!! pp_od_clk_voltage not found (needs amdgpu.ppfeaturemask in GRUB). Trying amd-smi for clocks only."
  [[ -n "$MAX_SCLK" ]] && { sudo amd-smi set -g "$GPU" -L sclk max "$MAX_SCLK" || true; }
  [[ -n "$MAX_MCLK" ]] && { sudo amd-smi set -g "$GPU" -L mclk max "$MAX_MCLK" || true; }
fi
amd-smi static -g "$GPU" --limit | grep -i "SOCKET_POWER_LIMIT" | head -1 || true
if [[ -n "$OD" ]]; then grep -A2 -E "OD_SCLK|OD_MCLK|OD_VDDGFX_OFFSET" "$OD" 2>/dev/null || true; fi
echo "Check during a big prompt read that the real clock stays at or below ${MAX_SCLK:-stock} MHz:"
echo "   watch -n 0.5 cat ${DEV:-/sys/class/drm/card1/device}/hwmon/hwmon*/freq1_input"

echo "== CPU settings =="
if [[ -n "$CPU_EPP" ]] && ls /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference >/dev/null 2>&1; then
  echo "$CPU_EPP" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference >/dev/null \
    && echo "CPU energy preference: $CPU_EPP" \
    || echo "!! CPU energy preference not applied."
fi
if [[ -n "$CPU_BOOST" && -e /sys/devices/system/cpu/cpufreq/boost ]]; then
  echo "$CPU_BOOST" | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null \
    && echo "CPU boost: $CPU_BOOST" \
    || echo "!! CPU boost setting not applied."
fi

echo "== Restarting Strata =="
strata launch --stop || true

echo "== Draft settings =="
set_engine_args

strata launch "${STRATA_ARGS[@]}"

echo "== Status =="
strata launch --status || true
