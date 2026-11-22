#!/usr/bin/env bash
# kbench18.sh — Q4_K branchless scales (STRATA_RDNA_KQ_SCB): bitwise check vs the stock wave kernel (RDNA_Q6K=0 makes native_mmvq the wave kernel), then 3 interleaved timing rounds SCB=1 / SCB=0
cd ~/work/rdna3
NAME=s18chk WHICH=mmvq KENV="STRATA_RDNA_Q6K=0 STRATA_RDNA_KQ_ALL=1" bash tools/kbench.sh
for i in 1 2 3; do
  NAME=s18new$i WHICH=mmvq KENV="STRATA_RDNA_KQ_ALL=1" bash tools/kbench.sh
  NAME=s18old$i WHICH=mmvq KENV="STRATA_RDNA_KQ_ALL=1 STRATA_RDNA_KQ_SCB=0" bash tools/kbench.sh
done
echo kbench18 done > logs/kbench18.txt
