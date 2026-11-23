#!/usr/bin/env bash
# wide_ab.sh — verify windows wider than 4 rows under the cumulative-probability rule (CUMX=1 decides each extra row against its ms): --spec 4 vs 6 vs 8 on the chat prompts, 3 alternated rounds, with the per-position draft trace
# usage (box, via setsid nohup): [ROWMS=1.1,2.0,2.9,3.6 ROUNDS=3] tools/wide_ab.sh   -> logs/wide-ab.txt   (rows past the list cost its last value)
cd ~/work/rdna3
OUT=logs/wide-ab.txt; R="STRATA_SPEC_CUMX_ROW_MS=${ROWMS:-1.1,2.0,2.9,3.6}"
: > $OUT
# s8x: the rule at 0.3x its threshold, so deep rows get verified even when it would cut them: slower, but its trace
# (logs/trace-s8x_*.txt) says what every stricter rule would have kept, without another run
ARMS="s4||;s6|$R|--spec 6;s8|$R|--spec 8;s8x|$R STRATA_SPEC_CUMX_SCALE=0.3|--spec 8" ROUNDS=${ROUNDS:-3} TRACE=1 OUT=$OUT bash tools/ab.sh
echo "wide_ab done" >> $OUT
