#!/usr/bin/env python3
# mkprompt.py — the bench's exact N-token prompt (rdna3/bench.py) or a free text, as token ids for the CLI engine (--tokens-file)
# usage (engine dir, venv): .venv/bin/python <repo>/tools/mkprompt.py <setup> <n|"text"> <out.txt>
import json, sys
from pathlib import Path
sys.path.insert(0, "rdna3")
import bench
from serve.frontend import openai_to_messages
cfg = json.loads(Path(f"rdna3/configs/{sys.argv[1]}.json").read_text()) if Path(f"rdna3/configs/{sys.argv[1]}.json").exists() else None
cfg = cfg or json.loads(Path(sys.argv[1]).read_text())
svc = bench.encoder(cfg)
arg = sys.argv[2]
text = bench.exact_prompt(svc, cfg.get("model_name", "m"), int(arg), 0) if arg.isdigit() else arg
ids = svc.encode_prompt(*openai_to_messages(bench.payload(cfg.get("model_name", "m"), text, 256)))
Path(sys.argv[3]).write_text(",".join(map(str, ids)))
print(len(ids), "tokens ->", sys.argv[3])
