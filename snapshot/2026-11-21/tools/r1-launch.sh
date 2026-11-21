#!/usr/bin/env bash
# r1-launch.sh — start the Strata engine WITHOUT touching the user's script, so it
# can be launched under a raised RLIMIT_MEMLOCK (the PLE table is 26.8 GiB and the
# engine log says locking it is worth up to 6% decode). Same arguments as
# start-strata.sh builds; clocks/CPU are applied separately with clocks2.sh.
# run it as: systemd-run --user --collect --remain-after-exit -p LimitMEMLOCK=infinity \
#              --working-directory=$HOME/work/strata --unit=strata-decode ./r1-launch.sh
export PATH=$HOME/.local/bin:$PATH
exec strata launch \
  --family qwen --model iq3_s \
  --context 128k --vision off \
  --drafting dflash2 --draft-vocab en --mtp off \
  --access local --yes --no-tui \
  --set STRATA_HIPBLASLT_TABLE=on
