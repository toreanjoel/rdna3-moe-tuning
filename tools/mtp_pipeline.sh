#!/usr/bin/env bash
# mtp_pipeline.sh — MTP head training data end to end on the box: prompts, greedy answers (normal server), replay through a dump server (STRATA_MTP_DUMP, no caches), then the normal server back
# usage: [T=data/mtp/train2 SRC=selfinstruct] tools/mtp_pipeline.sh [n_prompts] [max_tokens]    (resumable: gen skips done ids; the replay is redone only when the dump is missing)
cd ~/work/rdna3
N=${1:-1500}; MAXT=${2:-768}; T=${T:-data/mtp/train}; SRC=${SRC:-prompts}; mkdir -p $T
PY=engine/.venv/bin/python
ready() { for i in $(seq 1 120); do curl -sf -m 60 localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null && return 0; sleep 5; done; return 1; }
./strata-rdna3.sh stop >/dev/null 2>&1; ./strata-rdna3.sh serve >/dev/null 2>&1; ready || { echo "server did not come up"; exit 1; }
[ "$(wc -l < $T/prompts.jsonl 2>/dev/null || echo 0)" -ge $N ] || $PY tools/mtp_data.py $SRC $T/prompts.jsonl $N | awk 'NR % 50 == 0'
$PY tools/mtp_data.py gen $T/prompts.jsonl $T/gen.jsonl $MAXT 8080 | awk 'NR % 50 == 0'
if [ ! -s $T/dump.bin ]; then
  ./strata-rdna3.sh stop >/dev/null 2>&1
  XENV="STRATA_MTP_DUMP=$PWD/$T/dump.bin" XARGS="--prompt-cache 0 --conversation-cache-mib 0" ./strata-rdna3.sh serve >/dev/null 2>&1
  ready || { echo "dump server did not come up"; exit 1; }   # (the probe's few rows land in the dump too: harmless)
  $PY tools/mtp_data.py replay $T/gen.jsonl 8080
fi
./strata-rdna3.sh stop >/dev/null 2>&1; ./strata-rdna3.sh serve >/dev/null 2>&1
ls -la $T; echo "mtp_pipeline done"
