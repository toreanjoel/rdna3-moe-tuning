#!/usr/bin/env bash
# mtp_check_q4.sh — rerun tools/mtp_check.py on kept check data (KEEP=1 tools/mtp_check.sh) with the retrained weights and the run-time Q4_0 quantization, server stopped, then the server back
# usage (box, via setsid nohup): tools/mtp_check_q4.sh    -> logs/mtp-check-q4.txt
cd ~/work/rdna3
T=data/mtp/check
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
train-venv/bin/python tools/mtp_check.py --pre $T/pre.bin --dec $T/dec.bin --replay $T/replay.bin --ckpt data/mtp/ft/ckpt.pt --q4 \
  --vocab engine/rdna3/draft_vocab_zh60k.bin --rt-vocab data/mtp/rt-ft/draft_vocab.bin 2>&1 | grep -v Warning > logs/mtp-check-q4.txt
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "check done" >> logs/mtp-check-q4.txt
