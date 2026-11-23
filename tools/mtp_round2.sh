#!/usr/bin/env bash
# mtp_round2.sh — MTP head retrain round 2 on the box: diverse self-instruct prompts + real coding-session continuations (answers greedy and sampled, with and without thinking), prompt-path dumps, QAT training from the shipped weights, held-out evals (other prompts, other sessions), export, 3-round A/B against the current head
# usage (box, via setsid nohup): [N=2000 MAXT=384 STEPS=4000 VOCAB=engine/rdna3/draft_vocab_zh60k.bin] tools/mtp_round2.sh   -> logs/mtp-round2.txt
#        needs data/sessions/sess6-{train,test}.jsonl (tools/sessions.py --per 6) and data/mtp/check/replay.bin (KEEP=1 tools/mtp_check.sh)
#        writes data/mtp/{train2,train-sess,test-sess}/dump.bin (~20 KB a token: ~35 GB in all), data/mtp/ft2/ckpt.pt, data/mtp/rt-ft2
cd ~/work/rdna3
M=data/mtp; OUT=logs/mtp-round2.txt; N=${N:-2000}; MAXT=${MAXT:-384}; STEPS=${STEPS:-4000}
VOCAB=${VOCAB:-engine/rdna3/draft_vocab_zh60k.bin}; TP=train-venv/bin/python
{
date
for s in train test; do   # session requests are their own prompt files (the pipeline skips prompt writing when the file is there)
  mkdir -p $M/$s-sess; [ -s $M/$s-sess/prompts.jsonl ] || cp data/sessions/sess6-$s.jsonl $M/$s-sess/prompts.jsonl
  T=$M/$s-sess bash tools/mtp_pipeline.sh $(wc -l < $M/$s-sess/prompts.jsonl) $MAXT | tail -2
done
T=$M/train2 SRC=selfinstruct bash tools/mtp_pipeline.sh $N $MAXT | tail -2
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
date
mkdir -p $M/ft2
$TP tools/mtp_train.py train --dump $M/train/dump.bin,$M/train-sess/dump.bin,$M/train2/dump.bin --out $M/ft2/ckpt.pt --steps $STEPS \
  --chain 2 --vocab $VOCAB --qat --n-eval 60 --eval-every 1000 2>&1 | grep -v "Warning\|scaled_dot" | grep "eval\|base\|train /\|qat"
for d in $M/check/replay.bin $M/test-sess/dump.bin; do
  for ck in "" "--ckpt $M/ft/ckpt.pt" "--ckpt $M/ft2/ckpt.pt"; do
    echo "held out $d [${ck:-shipped}]: $($TP tools/mtp_train.py eval --dump $d --n-eval 100000 --qat --vocab $VOCAB $ck 2>&1 | grep 'eval on')"
  done
done
$TP tools/mtp_train.py export --ckpt $M/ft2/ckpt.pt --rt $M/rt-q4_0 --out $M/rt-ft2 | tail -1
date
} > $OUT 2>&1
[ -s $M/rt-ft2/dense.bin ] && ARMS="ft1||;ft2||--mtp $PWD/$M/rt-ft2" ROUNDS=3 TRACE=1 OUT=$OUT bash tools/ab.sh
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "mtp_round2 done" >> $OUT
