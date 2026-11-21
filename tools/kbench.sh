#!/usr/bin/env bash
# kbench.sh — isolated decode-kernel timings (engine's claude_bench, fused_gr_bench) with the production switches; stops the server for VRAM, restarts it after
# usage on the box: NAME=kb1 bash tools/kbench.sh   -> logs/kbench-kb1.txt
cd ~/work/rdna3
B=$PWD/engine/build-hip; OUT=logs/kbench-${NAME:-kb1}.txt
export HIP_VISIBLE_DEVICES=0 STRATA_GR_RDNA=1 STRATA_GR_RDNA_Q8=1 STRATA_GR_RDNA_A8=1 STRATA_GR_INJ_UP=1 STRATA_GR_FUSED=1 \
       STRATA_RDNA_MMVQ_WAVE=3 STRATA_RDNA_Q6K=1 ${KENV:-}
./strata-rdna3.sh stop >/dev/null 2>&1
{
if [ -n "${WHICH:-}" ]; then echo "== claude_bench $WHICH 0"; timeout 600 "$B/claude_bench" "$WHICH" 0 2>&1
else
for T in 1 2 3 4; do echo "== claude_bench all T$T"; timeout 300 "$B/claude_bench" all $T 2>&1 | tail -40; done
echo "== fused_gr_bench 200 1 4 8"; timeout 600 "$B/fused_gr_bench" 200 1 4 8 2>&1 | tail -40
fi
} > "$OUT" 2>&1
./strata-rdna3.sh serve >/dev/null 2>&1
echo "kbench done" >> "$OUT"
