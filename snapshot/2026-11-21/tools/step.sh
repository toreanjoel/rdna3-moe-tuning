#!/bin/bash
# step.sh — reversible steps toward 100+ tok/s on the 7900 XTX box (2026-11-21 plan).
# It NEVER restarts the server and never edits your start-strata.sh: it patches a copy
# (start-step.sh) and the model config, after snapshotting both into ~/work/strata/rollback/.
# You run the printed launch line yourself, so nothing goes down until you choose.
#   ./step.sh list | save | apply <id...> | status | probe [tokens] | rollback
set -uo pipefail
W=$HOME/work/strata
CFG=$(ls -t "$HOME"/.local/share/strata/install/strata-iq3_s.json 2>/dev/null | head -1)
LCFG=$HOME/.local/state/strata/launch/iq3_s-128k.json   # what the running server actually loads
SNAP=$W/rollback
STATE=$W/.step-state
log() { printf '%s\n' "$*"; }

snap() { # snap baseline | snap apply  — dir names are unique; a later snapshot can never
  # overwrite an earlier one (two applies in the same second used to clobber the baseline).
  mkdir -p "$SNAP"; local d; d=$(date +%Y%m%d-%H%M%S)-$$
  while [[ -e $SNAP/$d ]]; do d=$d-x; done
  mkdir -p "$SNAP/$d"
  cp -p "$W/start-strata.sh" "$SNAP/$d/"
  [[ -f $CFG ]] && cp -p "$CFG" "$SNAP/$d/strata-iq3_s.json"
  cp -p "$LCFG" "$SNAP/$d/" 2>/dev/null || true
  log "snapshot -> $SNAP/$d"
  # .step-last is the rollback baseline: set it on `save`, or on the FIRST apply only, so a
  # second apply (which snapshots already-patched configs) can never move it forward.
  if [[ "${1:-}" == baseline ]] || [[ ! -f $STATE ]]; then
    log "$SNAP/$d" > "$W/.step-last"; log "  ^ rollback baseline (pre-patch)"
  else
    log "  kept for audit only; rollback baseline unchanged"
  fi
}

ensure() {  # patches land on a copy; your start-strata.sh is never touched
  [[ -f $W/start-step.sh ]] || cp -p "$W/start-strata.sh" "$W/start-step.sh"
  log "patched copy: $W/start-step.sh (start-strata.sh untouched)"
}

flags() { # flags --flag value [--flag2 value ...]   -- with "" for a bare (valueless) flag
  python3 - "$CFG" "$LCFG" "$@" <<'PY'
import json, shutil, sys
paths, pairs = sys.argv[1:3], sys.argv[3:]

def put(a, flag, val):
    for i, x in enumerate(a):
        if x == flag:
            if val:
                a[i + 1] = val
            return
        if isinstance(x, str) and x.startswith(flag + "="):
            if val:
                a[i] = flag + "=" + val
            return
    a += [flag, val] if val else [flag]

def patch(p):
    shutil.copy2(p, p + ".stepbak")
    c = json.load(open(p))
    for key in ("args", "user_args"):
        a = c.get(key)
        if isinstance(a, list):
            for flag, val in zip(pairs[0::2], pairs[1::2]):
                put(a, flag, val)
    json.dump(c, open(p, "w"), indent=2)
    print("  %s -> %s (undo: cp %s.stepbak %s)" % (" ".join(pairs), p, p, p))

for path in paths:
    patch(path)
PY
}

addset() { # addset NAME=VALUE
  grep -q "$1" "$W/start-step.sh" && { log "  already set: $1"; return; }
  sed -i "0,/^STRATA_ENV=(/s//&\n  $1   # step.sh/" "$W/start-step.sh" && log "  --set $1"
}

setvar() { # setvar VAR value   (only in the copy)
  sed -i "s/^$1=.*/$1=$2   # step.sh/" "$W/start-step.sh" && log "  $1=$2"
}

case "${1:-list}" in
list)
  cat <<'EOT'
run ./step.sh apply <id...> in this order, one restart per step:
id         change                                          restart  root
memlock    LimitMEMLOCK=infinity on the systemd user mgr     yes     yes  (~6%, free watts)
calibrate  strata engine calibrate --measure (sets pcie/pool) yes    no  exclusive GPU 15-30 min
experts    --expert-cache 12000, --kv-resident 24576         yes     no  (model + launch config)
perlayer   --expert-cache-per-layer (bare flag; 2.97% -> 21.4%) yes   no  (both configs)
pcie       --pcie-frac 0.35 (or pcie=0.15)                   yes     no  (auto here is 0.55)
overlap    --set STRATA_ADAPT_OVERLAP=1                      yes     no
ccache     --conversation-cache-mib 4096 (keeps the PLE lock) yes    no  (both configs)
spec2      SPEC=3 -> SPEC=2 (verify step)                    yes     no
update     strata update -> engine 0.1.63 (RAM floor 17GiB)  yes     no  LAST, never mixed into an A/B
Note: EXP_B (./start-step.sh b) is a launch argument, not a patch — use it to test
prompt-read watts/whine, not decode speed.
EOT
  ;;
save) snap baseline ;;
apply)
  shift; snap apply
  for id in "$@"; do
    case $id in
      memlock)
        log "memlock (root, you run these 4 lines):"
        log '  sudo mkdir -p /etc/systemd/system/user@.service.d'
        log '  printf "[Service]\nLimitMEMLOCK=infinity\n" | sudo tee /etc/systemd/system/user@.service.d/60-strata-memlock.conf'
        log '  sudo systemctl daemon-reload && sudo systemctl restart user@$UID.service   # or reboot'
        log '  then: cd ~/work/strata && ./start-strata.sh   (reload takes ~17 min)'
        ;;
      calibrate)
        log "calibrate (its own run, no patch; the model is down for the sweep + reload):"
        log "  strata engine calibrate --profile launch-iq3_s-128k --measure"
        log "  strata engine calibrate --profile launch-iq3_s-128k --review-result"
        log "  strata engine calibrate --profile launch-iq3_s-128k --apply-result   # undo: --rollback-result"
        ;;
      perlayer) log "perlayer (bare flag, engine_help:224):"; ensure
        flags --expert-cache-per-layer
        ;;
      pcie|pcie=*) log "pcie:"; ensure
        val=${id#pcie=}; [[ $val == pcie ]] && val=0.35
        flags --pcie-frac "$val"     # STRATA_PCIE_FRAC_DEFAULT only takes on/off/old/auto
        ;;
      ccache|ccache=*) log "ccache:"; ensure
        val=${id#ccache=}; [[ $val == ccache ]] && val=4096
        flags --conversation-cache-mib "$val"   # keeps the PLE lock: 0.1.62 gives it back when RAM is tight
        ;;
      overlap) log "overlap:"; ensure; addset STRATA_ADAPT_OVERLAP=1 ;;
      spec2)   log "spec2:";   ensure; setvar SPEC 2 ;;
      experts) log "experts:"; ensure
        flags --expert-cache 12000 --kv-resident 24576
        ;;
      update)
        log "update (last; keeps the old engine dir for rollback):"
        log "  ls ~/.local/share/strata/components/engine/    # note 0.1.62-linux-x86_64-hip-rx7900"
        log "  strata update && strata --version"
        ;;
      *) log "unknown step: $id (see: ./step.sh list)" ;;
    esac
    echo "$id" >> "$STATE"
  done
  log ""
  log "when you are ready (drops the chat model for ~17 min):"
  log "  cd ~/work/strata && ./start-step.sh"
  log "then, once the server is ready, score it:"
  log "  python3 decode_watts.py 1000 112 280 step   # 1000-token decode, 280 W budget"
  ;;
status)
  pid=$(pgrep -f 'components/engine/.*/strata --serve' | head -1)
  [[ -n $pid ]] || { log "no engine running"; exit 1; }
  log "engine pid $pid  memlock: $(awk '/Max locked memory/{print $4}' /proc/$pid/limits) bytes"
  log "engine dir: $(tr '\0' '\n' < /proc/$pid/cmdline | grep -o 'components/engine/[^/]*' | head -1)"
  tr '\0' '\n' < /proc/$pid/environ | grep '^STRATA_' | sed 's/^/  env /' | head -20
  log "steps applied: $([ -f "$STATE" ] && tr '\n' ' ' < "$STATE" || echo none)"
  log "PLE table:"; grep -h "PLE table" "$HOME"/.local/state/strata/launch/*.engine.log 2>/dev/null | tail -2
  log "PCIe probe:"; grep -h "PCIe probe" "$HOME"/.local/state/strata/launch/iq3_s-128k.engine.log 2>/dev/null | tail -1
  ;;
probe)
  python3 "$W/decode_watts.py" "${2:-1000}" 112 280 "${3:-step}"
  python3 "$W/decode_probe.py" 2 "${2:-1000}" 112 "${3:-step}"
  ;;
rollback)
  d=$(cat "$W/.step-last" 2>/dev/null); [[ -d $d ]] || { log "no snapshot in $W/.step-last"; exit 1; }
  [[ -f $d/strata-iq3_s.json ]] && cp -p "$d/strata-iq3_s.json" "$CFG"
  if [[ -f $d/iq3_s-128k.json ]]; then
    cp -p "$d/iq3_s-128k.json" "$HOME/.local/state/strata/launch/iq3_s-128k.json"
    log "launch config restored (iq3_s-128k.json)"
  fi
  rm -f "$W/start-step.sh" "$STATE"
  log "config restored from $d, patched copy removed."
  log "then: cd ~/work/strata && ./start-strata.sh"
  ;;
*) sed -n '2,6p' "$0" ;;
esac
