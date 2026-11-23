#!/usr/bin/env bash
# mtp_check.sh — collect what tools/mtp_check.py needs on the box (decode-path + prompt-path rows of the chatbench prompts' answers, then the prompt-path replay of the same answers) and run it with the server stopped; the normal server comes back at the end
# usage (box, via setsid nohup): tools/mtp_check.sh [max_tokens]     -> logs/mtp-check.txt, data in data/mtp/check (deleted at the end unless KEEP=1)
cd ~/work/rdna3
T=data/mtp/check; MAXT=${1:-384}; OUT=logs/mtp-check.txt; PY=engine/.venv/bin/python
rm -rf $T; mkdir -p $T
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
{
$PY - <<'PY'
import json, re
src = open("tools/chatbench.py").read()
ps = eval(re.search(r"PROMPTS = (\[.*?\n\])", src, re.S).group(1))
ps += ["Write a bash script that backs up a directory to S3 with rotation, and explain each part.",
       "What causes inflation? Explain demand-pull and cost-push with examples.",
       "Implement an LRU cache in TypeScript with O(1) get and put, with comments.",
       "Summarise the causes and consequences of the French Revolution in about 300 words.",
       "Write SQL to find the top 3 customers by revenue per month, and explain window functions used.",
       "Describe how a B-tree index works and when a database chooses not to use it."]
open("data/mtp/check/prompts.jsonl", "w").write("".join(json.dumps({"id": i, "prompt": p}) + "\n" for i, p in enumerate(ps)))
PY
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
XENV="STRATA_MTP_DUMP=$PWD/$T/pre.bin STRATA_MTP_DUMP_DEC=$PWD/$T/dec.bin" XARGS="--prompt-cache 0 --conversation-cache-mib 0" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
ready || { echo "dump server did not come up"; exit 1; }
$PY tools/mtp_data.py gen $T/prompts.jsonl $T/gen.jsonl $MAXT 8080 | tail -1
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
XENV="STRATA_MTP_DUMP=$PWD/$T/replay.bin" XARGS="--prompt-cache 0 --conversation-cache-mib 0" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
ready || { echo "replay server did not come up"; exit 1; }
$PY tools/mtp_data.py replay $T/gen.jsonl 8080 | tail -1
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1
ls -la $T
for ck in "" "--ckpt data/mtp/ft/ckpt.pt"; do
  echo "== mtp_check ${ck:-shipped weights}"
  train-venv/bin/python tools/mtp_check.py --pre $T/pre.bin --dec $T/dec.bin --replay $T/replay.bin $ck \
    --vocab engine/rdna3/draft_vocab_zh60k.bin --rt-vocab data/mtp/rt-ft/draft_vocab.bin ${ck:+--q4} 2>&1 | grep -v Warning
done
} > $OUT 2>&1
[ "${KEEP:-0}" = 1 ] || rm -rf $T
./strata-rdna3.sh stop </dev/null >/dev/null 2>&1; ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "mtp_check done" >> $OUT
