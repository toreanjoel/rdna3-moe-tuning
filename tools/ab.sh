#!/usr/bin/env bash
# ab.sh — A/B arms on varied chats (tools/chatbench.py against a restarted server), ROUNDS alternated rounds; optional per-arm draft trace summary
# usage (box, via setsid nohup): ARMS="base||;cumx|STRATA_SPEC_CUMX=1|" [ROUNDS=3 OUT=logs/ab.txt TRACE=1 TIMING=1] tools/ab.sh
#        an arm is name|XENV|XARGS; TRACE=1 adds STRATA_SPEC_TRACE (logs/trace-<name>.txt, summarised with tools/spectrace.py);
#        TIMING=1 adds STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 and prints the server's decode lines per arm;
#        HITS=1 adds STRATA_DECODE_TIMING=1 only and prints the VRAM expert hit rate (tools/hitrate.py);
#        the answers of each arm go to logs/chat-<name>_<round>.json (compare two with tools/chatcmp.py);
#        SESS=file.jsonl benches session replays (tools/sessbench.py) instead of the chat prompts
cd ~/work/rdna3
OUT=${OUT:-logs/ab.txt}; ROUNDS=${ROUNDS:-3}
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
run() { local name=$1 env=$2 args=$3 tr=logs/trace-$1.txt
  ./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
  [ "${TRACE:-0}" = 1 ] && { rm -f $tr; env="$env STRATA_SPEC_TRACE=$PWD/$tr"; }
  [ "${TIMING:-0}" = 1 ] && env="$env STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1"
  [ "${HITS:-0}" = 1 ] && env="$env STRATA_DECODE_TIMING=1"
  XENV="$env" XARGS="$args" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
  ready || { echo "== $name did not come up"; return; }
  [ "${TRACE:-0}" = 1 ] && : > $tr   # drop the probe's rounds
  echo "== $name  XENV=[$2] XARGS=[$3]"
  local l0; l0=$(wc -l < logs/serve.log)
  if [ -n "${SESS:-}" ]; then (cd engine && .venv/bin/python ../tools/sessbench.py "$OLDPWD/$SESS" 8080 ${SESSTOK:-256} 2>&1 | tail -1)
  else (cd engine && CHATOUT=$OLDPWD/logs/chat-$name.json .venv/bin/python ../tools/chatbench.py 8080 ${CHATROUNDS:-1} 2>&1 | tail -1); fi
  [ "${HITS:-0}" = 1 ] && tail -n +$((l0 + 1)) logs/serve.log | python3 tools/hitrate.py | sed 's/^/   /'
  [ "${TRACE:-0}" = 1 ] && python3 tools/spectrace.py $tr | sed 's/^/   /'
  [ "${TIMING:-0}" = 1 ] && grep -a "decode GPU stages\|ms/window\|decode:" logs/serve.log | tail -4 | cut -c1-600 | sed 's/^/   /'
  return 0; }
{
for round in $(seq 1 $ROUNDS); do
  IFS=';' read -ra A <<< "$ARMS"
  for arm in "${A[@]}"; do IFS='|' read -r n e x <<< "$arm"; run "${n}_$round" "$e" "$x"; done
done
} >> $OUT 2>&1
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1; ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "ab done" >> $OUT
