#!/usr/bin/env bash
# quality.sh — is FAST=1 as good as the exact IQ3_S? Teacher-forced KL/top-1/perplexity + HumanEval pass@1, exact vs FAST. Run via ./strata-rdna3.sh quality
set -uo pipefail
: "${R:?}" "${SRC:?}" "${D:?}" "${SETUP:?}" "${PORT:?}" "${LAUNCHER:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$R/bench/quality-$(date +%Y%m%d-%H%M); mkdir -p "$OUT" "$D/eval"
HE=$D/eval/HumanEval.jsonl.gz
[ -s "$HE" ] || curl -fsSL -o "$HE" https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
up() { FAST=$1 "$LAUNCHER" serve </dev/null >/dev/null 2>&1
  for _ in $(seq 120); do curl -sf "localhost:$PORT/v1/models" >/dev/null && return 0; sleep 5; done; echo "server did not come up"; exit 1; }
kl() { local n=$1; shift; local ev=(); for kv in "$@"; do ev+=(--env "$kv"); done
  (cd "$SRC" && .venv/bin/python tools/quant_quality.py run --config "$SRC/rdna3/state/$SETUP-$PORT.json" \
     --out "$OUT/$n.tsv" --gpu-experts auto "${ev[@]}") > "$OUT/$n.log" 2>&1 || echo "KL run $n failed: $OUT/$n.log"; }
{
echo "== 1/2 next-token check: 5,000 tokens of code, docs, English, other languages, an agent transcript"
up 0; "$LAUNCHER" stop </dev/null >/dev/null 2>&1   # serve once so the resolved config exists, then free the GPU
kl exact
kl fast STRATA_ROUTE_TAIL_SKIP=7
(cd "$SRC" && .venv/bin/python tools/quant_quality.py compare "$OUT/exact.tsv" "$OUT/fast.tsv")
echo "   read: top1 = same next token as exact; PPL = perplexity of the real text (lower is better); KL 0 = identical"
echo "== 2/2 HumanEval (164 coding problems, greedy, tests run)"
up 0; python3 "$HERE/humaneval.py" --data "$HE" --out "$OUT/he-exact.json" --url "http://127.0.0.1:$PORT" | tail -1
up 1; python3 "$HERE/humaneval.py" --data "$HE" --out "$OUT/he-fast.json" --url "http://127.0.0.1:$PORT" | tail -1
python3 "$HERE/humaneval.py" --compare "$OUT/he-exact.json" "$OUT/he-fast.json"
echo "results: $OUT"
} 2>&1 | tee "$OUT/summary.txt"
up "${FAST:-1}"
