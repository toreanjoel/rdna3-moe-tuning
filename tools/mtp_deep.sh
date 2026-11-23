#!/usr/bin/env bash
# mtp_deep.sh — the deep-draft heads, in order of value: round 3 (teacher-labelled sessions + source files, chain 4: tools/mtp_round3.sh -> ft4), then its control, the same chain-4 training on round 2's data only (ft3), A/B'd against ft4 at --spec 4 and 8 (tokens kept a step is the target)
# usage (box, via setsid nohup, after tools/mtp_round2.sh): [CHAIN=4 CHAINW=0.6 ROWMS=1.1,2.0,2.9,3.6] tools/mtp_deep.sh   -> logs/mtp-round3.txt, logs/mtp-deep.txt, data/mtp/rt-ft4, rt-ft3
cd ~/work/rdna3
M=$PWD/data/mtp; R="STRATA_SPEC_CUMX_ROW_MS=${ROWMS:-1.1,2.0,2.9,3.6}"
[ -s data/sessions/win-train.jsonl ] && bash tools/mtp_round3.sh
B=ft2; [ -s $M/rt-ft4/dense.bin ] && B=ft4
# MTP_NOLAB=1: the control trains and scores as round 2 did (the text's own tokens as labels), only deeper
MTP_NOLAB=1 TAG=ft3 CHAIN=${CHAIN:-4} CHAINW=${CHAINW:-0.6} MTP_EVAL_STEPS=6 OUT=logs/mtp-deep.txt \
ABARMS="$B||--mtp $M/rt-$B;ft3||--mtp $M/rt-ft3;${B}w|$R|--mtp $M/rt-$B --spec 8;ft3w|$R|--mtp $M/rt-ft3 --spec 8" bash tools/mtp_round2.sh
