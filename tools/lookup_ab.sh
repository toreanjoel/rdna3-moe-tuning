#!/usr/bin/env bash
# lookup_ab.sh — prompt lookup chained after the MTP drafts (--lookup-chain K, off by default; the suffix drafter itself is on) on held-out coding sessions and on the chat prompts, 3 alternated rounds each
# usage (box, via setsid nohup): [K="3 5" ROUNDS=3] tools/lookup_ab.sh   (needs data/sessions/sess-test.jsonl from tools/sessions.py) -> logs/lookup-ab.txt
cd ~/work/rdna3
OUT=logs/lookup-ab.txt; A="base||"
for k in ${K:-3 5}; do A="$A;chain$k||--lookup-chain $k"; done
: > $OUT
ARMS="$A" ROUNDS=${ROUNDS:-3} HITS=1 SESS=data/sessions/sess-test.jsonl OUT=$OUT bash tools/ab.sh
ARMS="$A" ROUNDS=${ROUNDS:-3} HITS=1 OUT=$OUT bash tools/ab.sh
echo "lookup_ab done" >> $OUT
