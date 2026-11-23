#!/usr/bin/env bash
# mtp_smoke.sh — CPU smoke test of tools/mtp_train.py on the 12 check chats (safe beside a running server): label, eval with and without the backbone's labels, two training steps on a mixed (labelled + unlabelled) set
# usage (box, via setsid nohup): [SCRIPT=tools/mtp_train.py] tools/mtp_smoke.sh   -> logs/mtp-smoke.txt   (needs data/mtp/check/replay.bin and data/mtp/train/out_hc.pt from `prep`)
cd ~/work/rdna3
S=${SCRIPT:-tools/mtp_train.py}; V=tools/draft_vocab_plus.bin; TP="nice -n 19 train-venv/bin/python $S"
export MTP_DEV=cpu OMP_NUM_THREADS=6 MTP_EVAL_STEPS=6
{
cp data/mtp/check/replay.bin logs/smoke-dump.bin; rm -f logs/smoke-dump.bin.lab.npy
$TP label --dump logs/smoke-dump.bin 2>&1 | grep -v Warning | tail -2
echo "unlabelled: $($TP eval --dump data/mtp/check/replay.bin --n-eval 100000 --qat --vocab $V --ckpt data/mtp/ft/ckpt.pt 2>&1 | grep 'eval on')"
echo "labelled:   $($TP eval --dump logs/smoke-dump.bin --n-eval 100000 --qat --vocab $V --ckpt data/mtp/ft/ckpt.pt 2>&1 | grep 'eval on')"
timeout 3000 $TP train --dump data/mtp/check/replay.bin,logs/smoke-dump.bin --out logs/smoke-ckpt.pt --steps 2 --chain 3 --chain-w 0.6 --vocab $V --qat \
  --n-eval 2 --eval-every 100 2>&1 | grep -v Warning | tail -8
rm -f logs/smoke-dump.bin logs/smoke-dump.bin.lab.npy logs/smoke-ckpt.pt
echo "mtp_smoke done"
} > logs/mtp-smoke.txt 2>&1
