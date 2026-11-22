#!/usr/bin/env bash
# expbench.sh — grouped expert layouts (STRATA_EXP_MODE 0-9) on real IQ3_S rows, bitwise vs mode 0; server stopped for VRAM, restarted after
# usage on the box: NAME=e1 LAYERS=0,12,24,36,47 NG="8 24" TOKS="1 2" MODES="1 2 3 4 5 6 7 8 9" bash tools/expbench.sh
cd ~/work/rdna3
B=$PWD/engine/build-hip; OUT=logs/expbench-${NAME:-e1}.txt
G=$PWD/data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
export HIP_VISIBLE_DEVICES=0 ${KENV:-}
./strata-rdna3.sh stop >/dev/null 2>&1
{
for g in ${NG:-8 24}; do for t in ${TOKS:-1 2}; do for m in ${MODES:-1 2 3 4 5 6 7 8 9}; do
  echo "== groups $g tok/group $t mode $m"; timeout 300 "$B/native_expert_bench" "$G" "${LAYERS:-0,12,24,36,47}" $g $t 0 $m ${ITERS:-200} 2>&1 | tail -${TAILN:-12}
done; done; done
} > "$OUT" 2>&1
(./strata-rdna3.sh serve </dev/null >/dev/null 2>&1 &)
echo "expbench done" >> "$OUT"
