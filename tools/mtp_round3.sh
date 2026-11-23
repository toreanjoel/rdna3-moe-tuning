#!/usr/bin/env bash
# mtp_round3.sh — MTP head round 3 on the box: teacher-labelled data. Whole coding sessions and source files are replayed through the dump server (prompt path, ~1,800 tok/s, nothing generated) and every row is labelled with the backbone's own next token (mtp_train.py label), so all of that text trains the head; chain 4, scored to step 6, then an A/B at --spec 4 and 8
# usage (box, via setsid nohup, after tools/mtp_round2.sh): [STEPS=10000 CHAIN=4 CHAINW=0.6 TAG=ft4 FILECHARS=7000000 ROWMS=1.1,2.0,2.9,3.6 ABARMS=..] tools/mtp_round3.sh   -> logs/mtp-round3.txt, data/mtp/rt-ft4
#        needs data/sessions/win-{train,test}.jsonl (tools/sessions.py DIR PREFIX --windows --cap 16000) and round 2's dumps; ~20 KB of dump a token
cd ~/work/rdna3
M=data/mtp; OUT=${OUT:-logs/mtp-round3.txt}; TAG=${TAG:-ft4}; STEPS=${STEPS:-10000}; VOCAB=${VOCAB:-tools/draft_vocab_plus.bin}
TP=train-venv/bin/python; PY=engine/.venv/bin/python; export MTP_EVAL_STEPS=${MTP_EVAL_STEPS:-6}
R="STRATA_SPEC_CUMX_ROW_MS=${ROWMS:-1.1,2.0,2.9,3.6}"
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
dump() {   # $1/gen.jsonl (rows with their answer) -> $1/dump.bin
  [ -s $1/dump.bin ] && return 0
  ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  XENV="STRATA_MTP_DUMP=$PWD/$1/dump.bin" XARGS="--prompt-cache 0 --conversation-cache-mib 0" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
  ready || { echo "dump server did not come up"; return 1; }
  $PY tools/mtp_data.py replay $1/gen.jsonl 8080 | grep -c failed
  ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  ls -la $1/dump.bin; }
{
date
for s in train test; do mkdir -p $M/win-$s; [ -s $M/win-$s/gen.jsonl ] || cp data/sessions/win-$s.jsonl $M/win-$s/gen.jsonl; dump $M/win-$s; done
mkdir -p $M/files; [ -s $M/files/gen.jsonl ] || python3 tools/mtp_files.py $M/files/gen.jsonl ${FILECHARS:-7000000} engine ~/work/llama.cpp train-venv/lib
dump $M/files
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
date
[ -s $M/train/out_hc.pt ] || $TP tools/mtp_train.py prep --gguf $(ls data/models/IQ3_S/*00001-of-*.gguf) --out $M/train 2>&1 | tail -3
# round 1's dump stays unlabelled: all of it is this model's greedy text, whose own tokens are what the decode path chose
have() { local o=""; for d in "$@"; do [ -s $M/$d/dump.bin ] && o="$o,$M/$d/dump.bin"; done; echo "${o#,}"; }   # the dumps that exist
$TP tools/mtp_train.py label --dump $(have train-sess win-train files train2 win-test test-sess) 2>&1 | grep -v Warning
date
mkdir -p $M/$TAG
$TP tools/mtp_train.py train --dump $(have train train-sess win-train files train2) --out $M/$TAG/ckpt.pt \
  --steps $STEPS --chain ${CHAIN:-4} --chain-w ${CHAINW:-0.6} --vocab $VOCAB --qat --n-eval 60 --eval-every 2000 2>&1 | grep -v "Warning\|scaled_dot" | grep "eval\|base\|train /\|qat\|draws"
date
for d in $M/check/replay.bin $M/win-test/dump.bin $M/test-sess/dump.bin; do
  [ -s $d ] || continue
  for ck in "" ft ft2 ft3 $TAG; do
    [ -z "$ck" ] || [ -s $M/$ck/ckpt.pt ] || continue
    echo "held out $d [${ck:-shipped}]: $($TP tools/mtp_train.py eval --dump $d --n-eval 100000 --qat --vocab $VOCAB ${ck:+--ckpt $M/$ck/ckpt.pt} 2>&1 | grep 'eval on')"
  done
done
$TP tools/mtp_train.py export --ckpt $M/$TAG/ckpt.pt --rt $M/rt-q4_0 --out $M/rt-$TAG | tail -1
date
} > $OUT 2>&1
# the A/B: round 2's head, the deep-chain head if there is one, this head; the last two also with windows to 8 rows.
# logs/round3-arms.txt, if present when training ends, replaces the arms (a decision that can wait for the earlier A/Bs)
P=$PWD/$M; A="ft2||--mtp $P/rt-ft2"
[ -s $M/rt-ft3/dense.bin ] && A="$A;ft3||--mtp $P/rt-ft3;ft3w|$R|--mtp $P/rt-ft3 --spec 8" || A="$A;ft2w|$R|--mtp $P/rt-ft2 --spec 8"
A="$A;$TAG||--mtp $P/rt-$TAG;${TAG}w|$R|--mtp $P/rt-$TAG --spec 8"
[ -s logs/round3-arms.txt ] && A=$(cat logs/round3-arms.txt)
[ -s $M/rt-$TAG/dense.bin ] && ARMS="${ABARMS:-$A}" ROUNDS=3 TRACE=1 OUT=$OUT bash tools/ab.sh
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "mtp_round3 done" >> $OUT
