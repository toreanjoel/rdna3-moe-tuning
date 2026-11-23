# hitrate.py — VRAM expert hit rate and step cost from the server's "strata decode timing" lines (STRATA_DECODE_TIMING=1), read from stdin, weighted by windows
# usage: tail -n +N logs/serve.log | python3 tools/hitrate.py
import re, sys
w = hits = ent = ms = tok = rows = 0.0
pat = re.compile(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window.*?CPU experts [\d.]+ \(([\d.]+) entries\), VRAM hits ([\d.]+)")
for line in sys.stdin.buffer:
    m = pat.search(line.decode(errors="replace"))
    if not m: continue
    n = float(m.group(1)); w += n; rows += n * float(m.group(2)); tok += n * float(m.group(3)); ms += n * float(m.group(4))
    ent += n * float(m.group(5)); hits += n * float(m.group(6))
if not w: sys.exit("no decode timing lines")
print(f"{int(w)} windows: VRAM hit rate {hits / (hits + ent):.2%} ({hits / w:.2f} hits, {ent / w:.2f} CPU entries a layer-window), "
      f"{rows / w:.2f} rows, {tok / w:.2f} tokens, {ms / w:.2f} ms a window")
