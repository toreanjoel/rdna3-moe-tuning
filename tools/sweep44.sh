#!/usr/bin/env bash
# sweep44.sh — retrained MTP head (data/mtp/rt-ft, tools/mtp_train.py) vs the q4_0 head on varied chats (chatbench, 3 rounds alternated)
cd ~/work/rdna3
OUT=${OUT:-logs/sweep44.txt}
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
run() { ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  XENV="$2" XARGS="$3" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
  ready || { echo "== $1 did not come up"; return; }
  echo "== $1  XENV=[$2] XARGS=[$3]"
  (cd engine && .venv/bin/python ../tools/chatbench.py 8080 1 2>&1 | tail -1); }
{
for round in 1 2 3; do
run base_$round "" ""
run ft_$round "" "--mtp $PWD/data/mtp/rt-ft"
done
} >> $OUT 2>&1
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1; ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep44 done" >> $OUT
