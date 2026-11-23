#!/usr/bin/env bash
# profile_ab.sh — an expert profile learned from logged coding sessions (--expert-profile-save over data/sessions/sess-train.jsonl) vs the shipped one: VRAM hit rate and tok/s on held-out sessions and on the chat prompts
# usage (box, via setsid nohup): tools/profile_ab.sh     (needs data/sessions/sess-{train,test}.jsonl from tools/sessions.py) -> logs/profile-ab.txt
cd ~/work/rdna3
NEW=$PWD/data/expert-profile-coding.bin; OUT=logs/profile-ab.txt; S=data/sessions
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
{
if [ ! -s $NEW ]; then
  ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  XARGS="--expert-profile-save $NEW --expert-profile-save-every 1" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
  ready || { echo "learning server did not come up"; exit 1; }
  (cd engine && .venv/bin/python ../tools/sessbench.py ../$S/sess-train.jsonl 8080 256 | tail -1)
  sleep 70; ready; sleep 5   # a save happens between requests once a minute has passed
  ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  grep -a "profile" logs/serve.log | tail -3 | cut -c1-300
fi
ls -la $NEW engine/data/expert-profile.bin
} > $OUT 2>&1
[ -s $NEW ] || { echo "no profile was saved" >> $OUT; ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1; exit 1; }
ARMS="base||;coding||--expert-profile $NEW" ROUNDS=${ROUNDS:-3} HITS=1 SESS=$S/sess-test.jsonl OUT=$OUT bash tools/ab.sh
ARMS="base||;coding||--expert-profile $NEW" ROUNDS=${ROUNDS:-3} HITS=1 OUT=$OUT bash tools/ab.sh
echo "profile_ab done" >> $OUT
