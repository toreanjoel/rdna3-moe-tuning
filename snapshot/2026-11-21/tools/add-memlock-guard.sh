#!/bin/bash
# add-memlock-guard.sh — put the memlock guard in ~/work/strata/start-strata.sh (idempotent).
# The guard: prints a warning when the launching shell still carries the old 12063432 KB cap and
# tries `ulimit -l unlimited` (which works only if the hard limit allows it — true in a login
# session made after /etc/security/limits.d/60-strata-memlock.conf was added).
set -euo pipefail
W=$HOME/work/strata; F=$W/start-strata.sh
grep -qF 'MEMLOCK_OK' "$F" && { echo "guard already present"; exit 0; }
[[ -f $W/start-strata.sh.pre-memlock ]] || cp -p "$F" "$W/start-strata.sh.pre-memlock"
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]
lines = open(p).read().split("\n")
# undo a previous bad insertion (an unquoted heredoc wrote a literal "unlimited")
out, i = [], 0
while i < len(lines):
    if lines[i].startswith("# PLE table needs unlimited memlock"):
        while i < len(lines) and not lines[i].startswith("fi"):
            i += 1
        i += 1
        continue
    out.append(lines[i]); i += 1
lines = out
block = [
 "",
 "# PLE table needs unlimited memlock; a terminal opened before",
 "# /etc/security/limits.d/60-strata-memlock.conf still carries the old 12063432 KB cap, so the",
 "# engine would load for ~17 min with the PLE table unlocked. Raise first, then check the result.",
 "ulimit -l unlimited 2>/dev/null || true",
 'if [[ "$(ulimit -l)" != "unlimited" ]]; then',
 '  echo "!! memlock still $(ulimit -l) KB - open a NEW terminal (limits.d only applies to new logins) and run this again."',
 '  echo "   to load capped anyway: MEMLOCK_OK=1 ./start-strata.sh"',
 "  [[ ${MEMLOCK_OK:-} == 1 ]] || exit 1",
 "fi",
]
k = next(n for n, l in enumerate(lines) if l.startswith("set -euo pipefail"))
lines[k + 1:k + 1] = block
open(p, "w").write("\n".join(lines))
PY
bash -n "$F" && sed -n '/PLE table needs unlimited/,/ulimit -l unlimited/p' "$F"
sha256sum "$F" "$W/start-strata.sh.pre-memlock"
