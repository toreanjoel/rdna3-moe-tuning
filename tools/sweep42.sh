#!/usr/bin/env bash
# sweep42.sh — untested MTP drafter options on varied chats (chatbench, 1 round per serve, 2 rounds alternated): base (q4_0 head) / --mtp-hnorm stream / STRATA_MTP_FULL_HEAD=1 / en draft vocab
cd ~/work/rdna3
OUT=${OUT:-logs/sweep42.txt}
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
run() { ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  XENV="$2" XARGS="$3" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
  ready || { echo "== $1 did not come up"; return; }
  echo "== $1  XENV=[$2] XARGS=[$3]"
  (cd engine && .venv/bin/python ../tools/chatbench.py 8080 1 2>&1 | tail -1); }
{
for round in 1 2; do
run base_$round "" ""
run hnorm_$round "" "--mtp-hnorm stream"
run fullhead_$round "STRATA_MTP_FULL_HEAD=1" ""
run envocab_$round "" "--mtp-draft-vocab $PWD/engine/data/draft_vocab_en.bin"
done
} >> $OUT 2>&1
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1; ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep42 done" >> $OUT
