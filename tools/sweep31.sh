#!/usr/bin/env bash
# sweep31.sh — (RUNS="x1 s3 x2" QRUNS=x to redo parts) IQ3_XXS vs IQ3_S: waits for the XXS download, builds data/packs/iq3_xxs, adds configs/iq3xxs-1gpu.json, benches both (8K+32K, REP 3, daily tune) interleaved, then teacher-forced KL/top-1/PPL of each vs IQ3_S
cd ~/work/rdna3
OUT=${OUT:-logs/sweep31.txt}; E=$PWD/engine; D=$PWD/data
M=$D/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
{
until grep -q DONE logs/dl-iq3_xxs.log 2>/dev/null; do pgrep -f "IQ3_XXS-00001" >/dev/null || { echo "download stopped without DONE"; exit 1; }; sleep 60; done
echo "== sha256"; cat "$M.sha"; echo "expect 219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"
grep -q 219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15 "$M.sha" || { echo "sha mismatch"; exit 1; }

if [ ! -s "$D/packs/iq3_xxs/experts.bin" ]; then
  echo "== pack"; (cd "$E" && STRATA_GGUF_PY=$E/third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
     --gguf "$M" --out "$D/packs/iq3_xxs" --experts-bin) 2>&1 | tail -15
fi
du -sh "$D/packs/iq3_xxs"
sed -e 's#packs/iq3_s#packs/iq3_xxs#g' -e 's#models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-#models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-#' \
    -e 's#"qwen-flash-iq3"#"qwen-flash-iq3xxs"#' "$E/rdna3/configs/iq3-1gpu.json" > "$E/rdna3/configs/iq3xxs-1gpu.json"
grep -n -E "iq3_xxs|IQ3_XXS" "$E/rdna3/configs/iq3xxs-1gpu.json"

run() { NAME=$1 SETUP=$2 XENV="STRATA_DECODE_TIMING=1" REPEATS=${REP:-3} ./strata-rdna3.sh bench 8192,32768 </dev/null >/dev/null 2>&1
  echo "== $1 ($2)"; grep -A2 "input   prefill" logs/bench-$1.log | tail -2
  grep -h "expert cache [0-9]* slots" bench/$1/*/engine.log | head -1
  grep -h "ms/window = verify" bench/$1/*/engine.log | grep -o "avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window = verify [0-9.]*.*draft [0-9.]*; per layer-window: CPU experts [0-9.]* ([0-9.]* entries), VRAM hits [0-9.]*" | tail -2; }
for r in ${RUNS:-s1 x1 s2 x2}; do case $r in s*) run s31_$r iq3-1gpu;; x*) run s31_$r iq3xxs-1gpu;; esac; done

Q=$PWD/bench/quality31; mkdir -p "$Q"; ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
qrun() { (cd "$E" && .venv/bin/python tools/quant_quality.py run --config "$E/rdna3/state/$2-18087.json" --out "$Q/$1.tsv" --gpu-experts auto \
   --env STRATA_RDNA_PAIR=1 --env STRATA_GR_TAIL=1 --env STRATA_EXP_AUTO=1 --env STRATA_ROUTE_TAIL_SKIP=7) > "$Q/$1.log" 2>&1 || echo "$1 FAILED (see $Q/$1.log)"; }
for r in ${QRUNS:-s x}; do case $r in s) qrun s iq3-1gpu;; x) qrun x iq3xxs-1gpu;; esac; done
echo "== quality: XXS vs IQ3_S"; (cd "$E" && .venv/bin/python tools/quant_quality.py compare "$Q/s.tsv" "$Q/x.tsv")
} >> $OUT 2>&1
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep31 done" >> $OUT
