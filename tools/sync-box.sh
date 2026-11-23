#!/bin/bash
# sync-box.sh — push this repo's current (best) strata-rdna3.sh, patches/ and tools/ to the box so `serve` there is always the latest.
# usage: tools/sync-box.sh [user@host]   (default bonfire@10.0.0.51; leaves data/ engine/ logs/ bench/ alone)
set -euo pipefail
BOX=${1:-bonfire@10.0.0.51}; DEST=work/rdna3
here=$(cd "$(dirname "$0")/.." && pwd)
rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
  "$here/strata-rdna3.sh" "$here/patches" "$here/tools" "$BOX:$DEST/"
rev=$(git -C "$here" rev-parse --short HEAD); dirty=$(git -C "$here" status --porcelain | grep -q . && echo "+dirty" || true)
ssh -o BatchMode=yes "$BOX" "echo '$rev$dirty $(date -u +%FT%TZ)' > $DEST/REPO_REV"
echo "box $BOX:$DEST now at $rev$dirty"
