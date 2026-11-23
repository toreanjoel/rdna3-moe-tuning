#!/usr/bin/env bash
# mtp_deep.sh — a draft head trained 4 steps deep (chain 4) on round 2's data, scored to step 6, then A/B'd against round 2's head at --spec 4 and at --spec 8 under the cumulative rule (tokens kept a step is the target)
# usage (box, via setsid nohup, after tools/mtp_round2.sh): [CHAIN=4 CHAINW=0.6 ROWMS=1.1,2.0,2.9,3.6] tools/mtp_deep.sh   -> logs/mtp-deep.txt, data/mtp/rt-ft3
cd ~/work/rdna3
M=$PWD/data/mtp; R="STRATA_SPEC_CUMX_ROW_MS=${ROWMS:-1.1,2.0,2.9,3.6}"
TAG=ft3 CHAIN=${CHAIN:-4} CHAINW=${CHAINW:-0.6} MTP_EVAL_STEPS=6 OUT=logs/mtp-deep.txt \
ABARMS="ft2||--mtp $M/rt-ft2;ft3||--mtp $M/rt-ft3;ft2w|$R|--mtp $M/rt-ft2 --spec 8;ft3w|$R|--mtp $M/rt-ft3 --spec 8" bash tools/mtp_round2.sh
