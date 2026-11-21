#!/bin/bash
# vk-llama.sh — Route B: llama.cpp + Vulkan for the same model on this 7900 XTX box (2026-11-21).
# Nothing here touches Strata until the last command, and that command needs port 8080 free, so
# llama.cpp and Strata never run at once (8 GiB VRAM and the reverse-ssh tunnel both need the one port).
#   ./vk-llama.sh deps      install the Vulkan build deps (root, you run the printed sudo line)
#   ./vk-llama.sh build     clone + build llama.cpp with the Vulkan backend
#   ./vk-llama.sh info      compat test: loads the model and answers one token
#   ./vk-llama.sh bench     prompt + decode tok/s, no server, model loaded once
#   ./vk-llama.sh serve     run llama-server on 127.0.0.1:8080 (refuses while Strata holds the port)
set -uo pipefail
M=${M:-$HOME/.local/share/strata/install/qwen3.8-flash-next-iq3_s.gguf}
D=$HOME/work/vk/llama.cpp
BIN=$D/build/bin
MOE=${MOE:-24}            # experts kept in system RAM; sweep 24 -> 48 -> 99 (VRAM is only 8 GiB)
CTXT=${CTXT:-131072}

case "${1:-help}" in
deps)
  echo 'sudo apt install -y cmake libvulkan-dev vulkan-tools glslang-tools spirv-tools libglew-dev libvulkan1 mesa-vulkan-drivers'
  ;;
build)
  [[ -d $D ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$D"
  cd "$D"
  cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j"$(nproc)" -- llama-cli llama-server llama-bench
  echo "built into $BIN"
  ;;
info)
  [[ -f $M ]] || { echo "no GGUF at $M - download an Unsloth dynamic GGUF and set M=..."; exit 1; }
  # --info alone prints build info without loading the model, so generate one token for real.
  "$BIN/llama-cli" -m "$M" -ngl 99 --n-cpu-moe "$MOE" -no-cnv -p hi -n 1 2>&1 \
    | grep -iE "loading|load tensors|n_tensors|architecture|error|not supported|vulkan|hi" | head -20
  ;;
bench)
  "$BIN/llama-bench" -m "$M" -ngl 99 -ncmoe "$MOE" -p 512 -n 128 -r 2 2>&1 | tail -6
  ;;
serve)
  if ss -ltn 2>/dev/null | grep -q ':8080 '; then
    echo "port 8080 is still held by Strata - stop it first (Ctrl-C in its terminal), then re-run this."
    exit 1
  fi
  echo "llama-server on 127.0.0.1:8080 (your reverse-ssh tunnel keeps working unchanged)"
  exec "$BIN/llama-server" -m "$M" -ngl 99 --n-cpu-moe "$MOE" -c "$CTXT" -fa on \
       --host 127.0.0.1 --port 8080 --alias qwen3.8-flash-next
  ;;
*) sed -n '2,10p' "$0" ;;
esac
