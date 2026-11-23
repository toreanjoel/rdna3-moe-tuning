#!/usr/bin/env bash
# mix_ab.sh — q8_0 on only the most-used MTP head experts (rest at q4_0 values) vs all q4_0 vs all q8_0, with the retrained dense weights: tokens kept per round (draft trace) on the chat prompts; packs are emulated in the q8_0 layout (tools/mtp_mix.py), so tok/s is not the measure here
# usage (box, via setsid nohup): [HOT="32 64 128"] tools/mix_ab.sh     -> logs/mix-ab.txt; the emulated packs (2.7 GB each) are deleted at the end
cd ~/work/rdna3
M=data/mtp; OUT=logs/mix-ab.txt; HOT=${HOT:-"64 128"}
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
{
[ -s $M/usage.npy ] || train-venv/bin/python tools/mtp_mix.py usage --dump $M/train/dump.bin --ckpt $M/ft/ckpt.pt --out $M/usage.npy --n 200 2>&1 | grep -v Warning | tail -2
mkdir -p $M/rt-ftq8
for n in dense.bin dense.txt draft_vocab.bin; do ln -f $M/rt-ft/$n $M/rt-ftq8/$n; done
for n in experts.bin experts.txt; do ln -f $M/rt-q8_0/$n $M/rt-ftq8/$n; done
for h in $HOT; do engine/.venv/bin/python tools/mtp_mix.py pack --usage $M/usage.npy --hot $h --q4 $M/rt-q4_0 --q8 $M/rt-q8_0 --dense $M/rt-ft --out $M/rt-mix$h; done
} > $OUT 2>&1
ARMS="q4||;q8||--mtp $PWD/$M/rt-ftq8"
for h in $HOT; do ARMS="$ARMS;mix$h||--mtp $PWD/$M/rt-mix$h"; done
ARMS="$ARMS" ROUNDS=1 TRACE=1 OUT=$OUT bash tools/ab.sh
for h in $HOT; do rm -rf $M/rt-mix$h; done
rm -rf $M/rt-ftq8
echo "mix_ab done" >> $OUT
