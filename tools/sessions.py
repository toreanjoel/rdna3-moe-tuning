#!/usr/bin/env python3
# sessions.py — replay requests from logged coding-tool sessions (Claude Code transcripts, *.jsonl): each request is a slice of a real session (user turns, assistant text, tool calls, tool results) that ends where the assistant spoke next; for the expert-profile test (tools/profile_ab.sh)
# usage: sessions.py DIR OUT_PREFIX [--per 3] [--chars 28000] [--test 0.25]   -> OUT_PREFIX-train.jsonl, OUT_PREFIX-test.jsonl ({"id", "messages"}; whole sessions go to one side)
#        sessions.py DIR OUT_PREFIX --windows [--chars 40000]   whole sessions as consecutive windows, each ending on an assistant turn held in
#        "content" (the format tools/mtp_data.py replay reads): every token of a session, for the labelled MTP training set
#        the output holds private session text: keep it out of git (write it under .nullrouter/)
import argparse, glob, hashlib, json, os

def flat(path, cap):
    """-> [(role, text)] with tool calls / results rendered as text, consecutive same-role turns merged."""
    out = []
    for line in open(path, errors="replace"):
        try: e = json.loads(line)
        except ValueError: continue
        if e.get("type") not in ("user", "assistant") or e.get("isSidechain"): continue
        m = e.get("message") or {}
        c, role, parts = m.get("content"), m.get("role"), []
        if isinstance(c, str): parts.append(c)
        else:
            for b in c or []:
                t = b.get("type")
                if t == "text": parts.append(b.get("text", ""))
                elif t == "tool_use": parts.append("<tool_call>\n" + json.dumps({"name": b.get("name"), "arguments": b.get("input")}, ensure_ascii=False)[:cap] + "\n</tool_call>")
                elif t == "tool_result":
                    r = b.get("content")
                    if isinstance(r, list): r = "\n".join(x.get("text", "") for x in r if isinstance(x, dict))
                    parts.append("<tool_response>\n" + str(r or "")[:cap] + "\n</tool_response>")
        text = "\n".join(p for p in parts if p).strip()
        if not text or role not in ("user", "assistant"): continue
        if out and out[-1][0] == role: out[-1] = (role, out[-1][1] + "\n" + text)
        else: out.append((role, text))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("out"); ap.add_argument("--per", type=int, default=3); ap.add_argument("--chars", type=int, default=28000)
ap.add_argument("--test", type=float, default=0.25); ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--windows", action="store_true")
a = ap.parse_args()
sets = {"train": [], "test": []}
for path in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
    turns = flat(path, a.cap)
    if a.windows:
        side = "test" if int(hashlib.sha1(os.path.basename(path).encode()).hexdigest(), 16) % 100 < a.test * 100 else "train"
        msgs, n, k = [], 0, 0
        for r, t in turns:
            if not msgs and r != "user": continue
            msgs.append({"role": r, "content": t[:a.chars]}); n += len(msgs[-1]["content"])
            if r == "assistant" and n >= a.chars:
                sets[side].append({"id": f"{os.path.basename(path)[:8]}-w{k}", "messages": msgs[:-1], "content": msgs[-1]["content"]})
                msgs, n, k = [], 0, k + 1
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and n >= 2000:
            sets[side].append({"id": f"{os.path.basename(path)[:8]}-w{k}", "messages": msgs[:-1], "content": msgs[-1]["content"]})
        continue
    cuts = [i for i, (r, _) in enumerate(turns) if r == "assistant" and i >= 2]
    if not cuts: continue
    side = "test" if int(hashlib.sha1(os.path.basename(path).encode()).hexdigest(), 16) % 100 < a.test * 100 else "train"
    for k in range(a.per):   # evenly spread cut points: the request ends just before assistant turn `cut`
        cut = cuts[min(len(cuts) - 1, (2 * k + 1) * len(cuts) // (2 * a.per))]
        msgs, n = [], 0
        for r, t in reversed(turns[:cut]):
            if n + len(t) > a.chars and msgs: break
            msgs.append({"role": r, "content": t}); n += len(t)
        msgs.reverse()
        while msgs and msgs[0]["role"] != "user": msgs.pop(0)
        if len(msgs) >= 1 and msgs[-1]["role"] == "user":
            sets[side].append({"id": f"{os.path.basename(path)[:8]}-{cut}", "messages": msgs})
for side, rows in sets.items():
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen: seen.add(r["id"]); uniq.append(r)
    with open(f"{a.out}-{side}.jsonl", "w") as f:
        for r in uniq: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = sum(len(m['content']) for r in uniq for m in r['messages']) + sum(len(r.get('content', '')) for r in uniq)
    print(f"{side}: {len(uniq)} requests, {tot // max(1, len(uniq))} chars each on average, {tot / 1e6:.1f}M chars in all")
