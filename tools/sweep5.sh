#!/usr/bin/env bash
# sweep5.sh — untried engine switches on top of FAST=1 (greedy), plus coupled drafting at temperature 0.6; 5 repeats each; restarts the server after
cd ~/work/rdna3
run() { NAME=$1 XENV="$2" XARGS="$3" REPEATS=5 ./strata-rdna3.sh bench </dev/null >/dev/null 2>&1; echo "== $1  T=${BENCH_TEMP:-0} XENV=[$2] XARGS=[$3]"; grep -A3 "input   prefill" logs/bench-$1.log || tail -3 logs/bench-$1.log; }
{
run s5base   "" ""
run g1151    "STRATA_ATTN_LANECELL=1 STRATA_MMVF_ROWS=1 STRATA_TSUM=1 STRATA_LFUSE=1 STRATA_GDN_SPLIT=1 STRATA_PLE_BATCH=1 STRATA_EXPERT_V2=1" ""
run adaptlag "STRATA_ADAPT_LAG=2 STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_OVERLAP=0" ""
run copyprio "STRATA_COPY_HIP_PRIO=1" ""
run hostsib  "" "--host-core sibling"
run hcreq8   "STRATA_HC_REQ8=1" ""
export BENCH_TEMP=0.6
run t06base  "" ""
run t06coup  "STRATA_SPEC_COUPLED=1" ""
run t06gumb  "STRATA_SPEC_COUPLED=1 STRATA_SPEC_GUMBEL=1" ""
} > logs/sweep5.txt 2>&1
unset BENCH_TEMP
./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep5 done" >> logs/sweep5.txt
