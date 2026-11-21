#!/usr/bin/env bash
# sweep3.sh — drafting / launch-overhead levers (no expert skipping), 5 repeats each, GEMM table default on
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" XARGS="$3" REPEATS=5 ./strata-rdna3.sh bench </dev/null >/dev/null 2>&1; echo "== $1  XENV=[$2] XARGS=[$3]"; grep -A3 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
run s3base   "" ""
run minp03   "" "--spec-min-p 0.3"
run chain3   "" "--lookup-chain 3"
run draftall "STRATA_MTP_DRAFT_ALL=1" ""
run mtpfuse  "STRATA_MTP_QFUSE=1 STRATA_MTP_MULTICOPY=1 STRATA_MTP_GR_R8=1" ""
run mega     "STRATA_MEGAKERNEL=1" ""
} > logs/sweep3.txt 2>&1
