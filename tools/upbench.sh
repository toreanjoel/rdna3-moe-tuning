#!/usr/bin/env bash
# upbench.sh — A/B an upstream Strata engine bundle (default ~/work/strata/0.1.65) against ours on the same model, pack and config, via a shadow ROOT; our engine is never touched
# usage (on the box, server stopped by the script, restarted at the end): UP=~/work/strata/0.1.65 REP=3 bash tools/upbench.sh
set -u
cd ~/work/rdna3
R=$PWD; E=$R/engine; UP=${UP:-$HOME/work/strata/0.1.65}; S=$UP/root
LEN=${LEN:-8192,32768}; REP=${REP:-3}; ROUNDS=${ROUNDS:-1}
./strata-rdna3.sh patch </dev/null >/dev/null 2>&1   # our config as `serve` would run it
# shadow ROOT: everything links to our tree except engine/ (upstream binary + its libs) and rdna3/ (real copies so ROOT resolves here)
rm -rf "$S"; mkdir -p "$S/engine" "$S/rdna3"
for f in "$E"/* "$E"/.venv "$E"/.config; do b=$(basename "$f"); case $b in engine|rdna3) ;; *) ln -s "$f" "$S/$b";; esac; done
for f in "$E"/rdna3/*; do b=$(basename "$f"); case $b in run.py|bench.py|configs|state|__pycache__) ;; *) ln -s "$f" "$S/rdna3/$b";; esac; done
cp "$E"/rdna3/run.py "$E"/rdna3/bench.py "$S/rdna3/"; cp -r "$E"/rdna3/configs "$S/rdna3/"
ln -s "$UP/strata" "$S/engine/strata"
printf '{"source":"upstream","backend":"hip","version":"%s","archs":["gfx1100"],"vision":"none","lib_dirs":["%s"]}\n' \
  "$(python3 -c "import json;print(json.load(open('$UP/BUILD.json'))['version'])")" "$UP" > "$S/engine/BUILD.json"
# upstream lacks --vram-reserve-mib/--mtp-window/--pcie-frac (fork-only); use its own hipBLASLt table (matches its bundled lib)
python3 - "$S/rdna3/configs/iq3-1gpu.json" "$UP" <<'EOF'
import json, sys, glob
p, up = sys.argv[1], sys.argv[2]
c = json.load(open(p)); a = c["args"]; out = []; i = 0
drop = {"--vram-reserve-mib", "--mtp-window", "--pcie-frac"}
while i < len(a):
    if a[i] in drop: i += 2; continue
    out.append(a[i]); i += 1
c["args"] = out
# the fork's STRATA_* tuning env crashes upstream (fused_gr_read_multi during capture): run upstream's own defaults
c["env"] = {}
t = glob.glob(up + "/tuning/gfx1100-hipblaslt-*.txt")
if t: c["env"]["STRATA_HIPBLASLT_TUNING"] = t[0]
json.dump(c, open(p, "w"), indent=1)
EOF
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
{
for i in $(seq "$ROUNDS"); do
  [ "${SKIP_OURS:-0}" = 1 ] || {
  NAME=upA_ours$i REPEATS=$REP ./strata-rdna3.sh bench "$LEN" </dev/null >/dev/null 2>&1
  echo "== ours (round $i)"; grep -A2 "input   prefill" logs/bench-upA_ours$i.log | tail -2; }
  # same FAST (route tail skip 7) and 128K context as ours
  (cd "$S" && STRATA_X_ENV="STRATA_ROUTE_TAIL_SKIP=7" STRATA_X_ARGS="--max-context 131072" .venv/bin/python rdna3/bench.py iq3-1gpu --data "$R/data" --gpus 0 --lengths "$LEN" --repeats "$REP" \
     --out "$R/bench/upA_up$i") > logs/bench-upA_up$i.log 2>&1
  echo "== upstream $(basename "$UP") (round $i)"; grep -A2 "input   prefill" logs/bench-upA_up$i.log | tail -2 || tail -5 logs/bench-upA_up$i.log
done
} > logs/upbench.txt 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "upbench done" >> logs/upbench.txt
