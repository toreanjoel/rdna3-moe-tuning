#!/usr/bin/env bash
# strata-rdna3.sh — install, run and benchmark the RDNA3 Strata fork (nicejunjie/Strata-rdna3) on one RX 7900 XTX.
# Usage: ./strata-rdna3.sh install | serve | stop | status | monitor | speed | tune | patch | bench [lengths] | uninstall
# Env: FAST HOST PORT SETUP GPUS TUNE SCLK MCLK VO WATTS EPP BOOST, XENV/XARGS (engine overrides), NAME/REPEATS (bench)
set -euo pipefail

REPO=https://github.com/nicejunjie/Strata-rdna3
PIN=${PIN:-6c2cd28b96c36b9acba87b4a54829ac5562d1a7c}   # main on 2026-11-20, measured 2026-11-21
R=${R:-$HOME/work/rdna3}                               # everything of the fork lives here
SRC=${SRC:-$R/engine}                  # engine source, build, .venv
D=${D:-$R/data}                        # models/ packs/ mtp/ for the fork
OLD=${OLD:-$HOME/.local/share/strata/Strata-data}      # the stock Strata data folder (models reused, never moved)
SETUP=${SETUP:-iq3-1gpu}                               # rdna3/configs/<SETUP>.json
GPUS=${GPUS:-0}                                        # HIP index of the 7900 XTX (the iGPU is 1)
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
LOG=${LOG:-$R/logs/serve.log}
MODEL=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
mkdir -p "$R/logs"
# engine overrides on top of rdna3/configs/$SETUP.json, e.g. XENV="STRATA_AUX_CPUS=auto" XARGS="--pcie-frac 0.15"
# hipBLASLt table matching ROCm 7.2.x (runtime 100202); the shipped config names 100200, which the engine rejects
# and falls back to untuned GEMMs. Measured 2026-11-21: prompts 2641->2840 (8K), 3266->3553 (32K); decode unchanged.
TABLE=${TABLE:-$SRC/tools/hip/gfx1100-hipblaslt-100202.txt}
# FAST=1: skip a missed expert that every token of a verify window ranks 8th-10th (STRATA_ROUTE_TAIL_SKIP=7).
# Measured 2026-11-22: decode 101->111 (8K), 103->109 (32K); answers change (top-1 agreement 92.6%, KL 0.050,
# perplexity of the reference text 4.064->4.041). Off by default: the default gives the engine's exact answers.
FASTENV=""; [ "${FAST:-0}" = 1 ] && FASTENV="STRATA_ROUTE_TAIL_SKIP=7"
export STRATA_X_ENV="STRATA_HIPBLASLT_TUNING=$TABLE $FASTENV ${XENV:-}" STRATA_X_ARGS="${XARGS:-}"

# setup.py moves the models of every Strata install listed in ~/.config/strata; a private config stops that.
export XDG_CONFIG_HOME=$SRC/.config

say() { printf '\n== %s\n' "$*"; }
port_busy() { ss -ltn | grep -q ":$PORT "; }

stop_all() {
  pkill -TERM -f '[r]dna3/run.py' || true
  pkill -TERM -f '[e]ngine/strata --serve' || true
  pkill -TERM -f '[s]erve/server.py --engine strata' || true
  command -v strata >/dev/null && strata launch --stop >/dev/null 2>&1 || true
  ~/.local/bin/strata launch --stop >/dev/null 2>&1 || true
  pkill -TERM -f '[s]trata --serve --pack' || true
  for _ in $(seq 60); do port_busy || return 0; sleep 1; done
  echo "port $PORT still busy"; return 1
}

install() {
  say "system packages (sudo)"
  sudo apt install -y build-essential git cmake ninja-build python3-venv
  [ -d /opt/rocm ] || { echo "ROCm not found in /opt/rocm: install ROCm 7.x first (amdgpu-install --usecase=rocm)"; exit 1; }
  groups | grep -qw render || echo "warning: $USER is not in the render group (sudo usermod -aG render,video $USER, then log in again)"

  say "source at $PIN"
  [ -d "$SRC/.git" ] || git clone "$REPO" "$SRC"
  git -C "$SRC" fetch --depth 1 origin "$PIN" 2>/dev/null || git -C "$SRC" fetch origin
  git -C "$SRC" checkout -q "$PIN"

  say "model"
  mkdir -p "$D/models"
  if [ -e "$OLD/models/IQ3_S/$MODEL" ] && [ ! -e "$D/models/IQ3_S" ]; then
    ln -sfn "$OLD/models/IQ3_S" "$D/models/IQ3_S"; echo "reusing $OLD/models/IQ3_S (symlink, nothing copied)"
  elif [ ! -e "$D/models/IQ3_S/$MODEL" ]; then
    echo "no IQ3_S found: setup will download it (~82 GB) into $D/models/IQ3_S"
  fi

  say "toolchain + engine build + MTP head (10-20 min)"
  (cd "$SRC" && ./setup.sh --setup --build --backend hip --family qwen --model IQ3_S --vision none \
     --data-dir "$D" --no-start --yes) 2>&1 | tee "$R/logs/setup.log"

  say "expert pack (~47 GB, once)"
  if [ ! -s "$D/packs/iq3_s/experts.bin" ]; then
    (cd "$SRC" && STRATA_GGUF_PY=$SRC/third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
       --gguf "$D/models/IQ3_S/$MODEL" --out "$D/packs/iq3_s" --experts-bin)
  fi

  patch_fork
  say "done"; du -sh "$SRC" "$D"/packs "$D"/mtp 2>/dev/null
}

# Local patches to the fork (idempotent): bench.py without vision; run.py takes overrides from XENV / XARGS.
patch_fork() {
  sed -i 's/rdna3_run.config(a.setup, a.data, a.gpus, vision=True)/rdna3_run.config(a.setup, a.data, a.gpus, vision=False)/;
          s|"--log", str(out / "engine.log")\]|"--log", str(out / "engine.log"), "--no-vision"]|' "$SRC/rdna3/bench.py"
  grep -q 'STRATA_X_ENV' "$SRC/rdna3/run.py" && return
  "$SRC/.venv/bin/python" - "$SRC/rdna3/run.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = '    cfg = fill(cfg, {"@STRATA@": str(ROOT), "@DATA@": str(data)})\n'
hook = anchor + '''    # local patch (strata-rdna3.sh): STRATA_X_ENV="K=V ..." sets env; STRATA_X_ARGS="--flag v --switch" sets args
    import shlex
    for kv in shlex.split(os.environ.get("STRATA_X_ENV", "")):
        k, _, v = kv.partition("="); cfg["env"][k] = v
    xa = shlex.split(os.environ.get("STRATA_X_ARGS", "")); i = 0
    while i < len(xa):
        f = xa[i]; v = xa[i + 1] if i + 1 < len(xa) and not xa[i + 1].startswith("--") else None
        if f in cfg["args"]:
            j = cfg["args"].index(f)
            if v is not None and j + 1 < len(cfg["args"]) and not cfg["args"][j + 1].startswith("--"): cfg["args"][j + 1] = v
            elif v is not None: cfg["args"].insert(j + 1, v)
        else:
            cfg["args"] += [f] + ([v] if v is not None else [])
        i += 2 if v is not None else 1
'''
assert anchor in s, "run.py changed upstream; patch by hand"
if "import os" not in s: s = s.replace("import json", "import json\nimport os", 1)
p.write_text(s.replace(anchor, hook, 1))
PY
}

# GPU/CPU tuning from start-strata.sh (measured 2026-11-21): <=280-294 W, no SCLK max (coil whine), VRAM 1300.
SCLK=${SCLK:-2300}; MCLK=${MCLK:-1300}; VO=${VO:--50}; WATTS=${WATTS:-294}; EPP=${EPP:-performance}; BOOST=${BOOST:-1}

card() { for c in /sys/class/drm/card*/device; do [ "$(cat "$c/device" 2>/dev/null)" = 0x744c ] && { echo "$c"; return; }; done; }

tune() {
  local dev od hw want have
  dev=$(card); [ -n "$dev" ] || { echo "7900 XTX (0x744c) not found"; return 1; }
  od=$dev/pp_od_clk_voltage; hw=$(ls -d "$dev"/hwmon/hwmon* | head -1)
  [ -e "$od" ] || { echo "no $od: add amdgpu.ppfeaturemask=0xffffffff to GRUB"; return 1; }
  want="$SCLK $MCLK $VO $WATTS $EPP $BOOST"
  have="$(grep -A2 OD_SCLK "$od" | sed -n 's/^1: *\([0-9]*\).*/\1/p') $(grep -A2 OD_MCLK "$od" | sed -n 's/^1: *\([0-9]*\).*/\1/p') \
$(grep -A1 OD_VDDGFX_OFFSET "$od" | tail -1 | tr -dc '0-9-') $(( $(cat "$hw/power1_cap") / 1000000 )) \
$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null) $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
  if [ "$(echo $have)" = "$want" ]; then echo "tuning already applied: sclk $SCLK mclk $MCLK vo $VO ${WATTS}W epp $EPP boost $BOOST"; return; fi
  echo "applying sclk $SCLK mclk $MCLK vo $VO ${WATTS}W epp $EPP boost $BOOST (was: $have)"
  # one sudo shell: separate sudo tee calls lose OD writes on this box
  sudo bash -c "echo on > $dev/power/control; echo r > $od; echo c > $od
    echo 's 1 $SCLK' > $od; echo 'm 1 $MCLK' > $od; [ '$VO' = 0 ] || echo 'vo $VO' > $od; echo c > $od
    echo $(( WATTS * 1000000 )) > $hw/power1_cap
    for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo $EPP > \$f; done
    [ -e /sys/devices/system/cpu/cpufreq/boost ] && echo $BOOST > /sys/devices/system/cpu/cpufreq/boost; true"
  grep -A2 -E 'OD_SCLK|OD_MCLK|OD_VDDGFX' "$od" | tr '\n' ' '; echo
}

serve() {
  [ "${TUNE:-1}" = 1 ] && tune
  stop_all
  cd "$SRC"
  nohup .venv/bin/python rdna3/run.py "$SETUP" --data "$D" --gpus "$GPUS" --no-vision \
    --host "$HOST" --port "$PORT" --log "$LOG" > "${LOG%.log}.out" 2>&1 &
  for _ in $(seq 180); do port_busy && break; sleep 1; done
  port_busy && echo "serving http://$HOST:$PORT/v1 (log $LOG)" || { echo "did not start; see ${LOG%.log}.out"; exit 1; }
}

status() {
  port_busy && echo "port $PORT: up" || echo "port $PORT: down"
  pgrep -af '[e]ngine/strata --serve' | cut -c1-120 || true
  rocm-smi --showuse --showpower --showmeminfo vram 2>/dev/null | grep -E 'GPU\[0\]' || true
}

bench() {
  [ "${TUNE:-1}" = 1 ] && tune
  stop_all
  cd "$SRC"
  .venv/bin/python rdna3/bench.py "$SETUP" --data "$D" --gpus "$GPUS" --lengths "${1:-8192,32768}" \
    --repeats "${REPEATS:-3}" --out "$R/bench/${NAME:-default}" 2>&1 | tee "$R/logs/bench-${NAME:-default}.log"
}

uninstall() {
  stop_all
  echo "removing $SRC and $D (the stock Strata folder $OLD is not touched)"
  rm -rf "$SRC" "$D"
}

case "${1:-}" in
  install) install ;;
  serve) serve ;;
  stop) stop_all ;;
  monitor) exec "$SRC/bin/strata" monitor --url "http://127.0.0.1:$PORT" ;;   # live TUI; q detaches, server keeps running
  speed) tail -n "${N:-10}" -F "$LOG" | grep --line-buffered -oE 'prompt [0-9]+ tokens.*tok/s\)|drafts accepted [0-9]+ of [0-9]+' ;;
  status) status ;;
  tune) tune ;;
  patch) patch_fork ;;
  bench) bench "${2:-}" ;;
  uninstall) uninstall ;;
  *) sed -n 2,3p "$0"; exit 1 ;;
esac
