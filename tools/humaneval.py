#!/usr/bin/env python3
# humaneval.py — pass@1 (greedy) on HumanEval's 164 problems through the running server's OpenAI API; writes per-problem results.
# usage: python3 tools/humaneval.py --data HumanEval.jsonl.gz --out RUN.json [--url http://127.0.0.1:8080] [--effort low]
#        python3 tools/humaneval.py --compare A.json B.json
# Model code runs in a throwaway temp dir, isolated interpreter, 15 s timeout per problem.
import argparse, gzip, json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def ask(url, prompt, effort, max_tokens):
    body = {"messages": [{"role": "user", "content":
            "Complete this Python function. Reply with the whole function (with its imports) in one ```python block.\n\n"
            "```python\n" + prompt + "```"}],
            "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": effort}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


def extract(text, prompt):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    # keep the prompt's imports/helpers when the answer only gives the function
    return code if "import" in code or "def " not in prompt.split("def ")[0] else prompt.split("def ")[0] + code


def run_test(code, prob):
    src = code + "\n\n" + prob["test"] + f"\n\ncheck({prob['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=15)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--effort", default="low"); ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        A, B = (json.load(open(p)) for p in a.compare)
        pa, pb = {r["task_id"]: r["pass"] for r in A["results"]}, {r["task_id"]: r["pass"] for r in B["results"]}
        both = sorted(set(pa) & set(pb))
        sa, sb = sum(pa[t] for t in both), sum(pb[t] for t in both)
        only_a = [t for t in both if pa[t] and not pb[t]]; only_b = [t for t in both if pb[t] and not pa[t]]
        print(f"HumanEval pass@1: {os.path.basename(a.compare[0])} {sa}/{len(both)}  {os.path.basename(a.compare[1])} "
              f"{sb}/{len(both)}  (only first: {len(only_a)}, only second: {len(only_b)}; a gap of 3-4 is run-to-run noise)")
        return
    probs = [json.loads(l) for l in gzip.open(a.data, "rt")]
    t0 = time.time()

    def one(p):
        text, usage = ask(a.url.rstrip("/"), p["prompt"], a.effort, a.max_tokens)
        ok = run_test(extract(text, p["prompt"]), p)
        return {"task_id": p["task_id"], "pass": ok, "tokens": usage.get("completion_tokens")}

    res = []
    with ThreadPoolExecutor(2) as ex:   # one ask in flight + one test running
        for i, r in enumerate(ex.map(one, probs), 1):
            res.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(probs)}  pass {sum(x['pass'] for x in res)}  {time.time() - t0:.0f}s", flush=True)
    n = sum(r["pass"] for r in res)
    json.dump({"pass": n, "total": len(res), "effort": a.effort, "seconds": round(time.time() - t0), "results": res},
              open(a.out, "w"), indent=1)
    print(f"pass@1 {n}/{len(res)}  ({time.time() - t0:.0f}s)")


main()
