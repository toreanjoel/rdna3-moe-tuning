#!/usr/bin/env python3
# mtp_data.py — MTP head training data: varied prompts, the model's own greedy answers, then a replay through a server started with STRATA_MTP_DUMP (final residual rows + tokens)
# usage: mtp_data.py prompts OUT.jsonl [N]          deterministic varied prompts (code-heavy, prose, Q&A, lists)
#        mtp_data.py gen PROMPTS.jsonl OUT.jsonl [max_tokens] [port]   greedy answers (resumable: skips done ids)
#        mtp_data.py replay GEN.jsonl [port] [start] [end]             [user, assistant(reasoning+content)] -> 1 token, so the
#                                                     prompt path dumps every row (start the server with STRATA_MTP_DUMP=<file>
#                                                     and --conversation-cache-mib 0)
import json, random, sys, time, urllib.request

LANGS = ["Python", "TypeScript", "Rust", "Go", "C++", "Bash", "SQL", "Java", "C", "Elixir"]
CODE = [
    "Write a {l} function that {t}. Include a short usage example.",
    "Here is a bug report: my {l} code that {t} gives wrong results for empty input. Show a corrected version and explain the fix.",
    "Refactor a {l} implementation that {t} so it is easier to test. Explain the design.",
    "Write unit tests in {l} for a function that {t}.",
    "What is the time and space complexity of the usual way to write code that {t} in {l}? Show the code.",
    "Review this plan: a {l} service endpoint that {t}. What edge cases and failure modes should I handle?",
]
TASKS = ["parses an ISO-8601 date string", "merges overlapping intervals", "deduplicates a list while keeping order",
         "computes a rolling average over a stream", "implements an LRU cache", "validates an email address",
         "reads a CSV file and groups rows by a column", "retries an HTTP request with exponential backoff",
         "finds the k most frequent words in a text", "serializes a tree to JSON and back", "throttles function calls",
         "implements binary search on a rotated array", "walks a directory and sums file sizes", "parses command line flags",
         "builds a trie for autocomplete", "detects a cycle in a directed graph", "converts Roman numerals to integers",
         "implements a rate limiter with a token bucket", "diffs two lists of records by id", "paginates a database query",
         "computes Levenshtein distance", "schedules tasks with dependencies (topological sort)", "streams a large file in chunks",
         "implements a simple pub/sub event bus", "formats a number of bytes as a human readable string",
         "reverses a linked list", "checks whether a string is a valid parenthesis sequence", "flattens nested dictionaries",
         "computes shortest paths with Dijkstra", "hashes passwords safely"]
TOPICS = ["photosynthesis", "the French Revolution", "how TCP congestion control works", "inflation and interest rates",
          "black holes", "the immune system", "public key cryptography", "the history of the printing press",
          "how vaccines are tested", "plate tectonics", "how GPUs differ from CPUs", "the causes of World War I",
          "sourdough fermentation", "how a compiler works", "climate feedback loops", "the stock market",
          "memory hierarchies in computers", "how neural networks learn", "the water cycle", "Roman engineering",
          "sleep and memory", "how batteries store energy", "the Silk Road", "quantum entanglement", "game theory",
          "how DNS resolution works", "container orchestration", "the origins of jazz", "coral reef ecosystems",
          "the design of relational databases"]
PROSE = [
    "Explain {t} to a curious high school student.", "Give me a concise summary of {t} in five bullet points.",
    "What are the most common misconceptions about {t}?", "Write a short blog post introduction about {t}.",
    "Compare two competing explanations or approaches related to {t}.", "Write three quiz questions with answers about {t}.",
    "I have to give a two-minute talk on {t}. Draft it.", "List the key terms someone should know about {t}, with one-line definitions.",
]
MISC = ["Write a polite email declining a meeting invitation and proposing next week instead.",
        "Plan a three-day trip to Lisbon on a moderate budget.", "Write a short story about a lighthouse keeper who finds a radio.",
        "Give me a weekly workout plan for a beginner with no equipment.", "Draft a product description for a reusable water bottle.",
        "How should I prepare for a system design interview?", "Write a haiku sequence about autumn in a city.",
        "Explain the difference between a mutex and a semaphore with an analogy.", "Help me write a cover letter for a junior data analyst role.",
        "What are good practices for writing commit messages?", "Translate 'The meeting has been moved to Thursday afternoon' into Spanish, French and German.",
        "Summarize the pros and cons of remote work.", "Create a meal plan for a vegetarian week.", "Write a limerick about a cat who codes."]

def prompts(out, n):
    rnd = random.Random(1234)
    seen, rows, tries = set(), [], 0
    while len(rows) < n and tries < 200000:
        tries += 1
        r = rnd.random()
        if r < 0.55: p = rnd.choice(CODE).format(l=rnd.choice(LANGS), t=rnd.choice(TASKS))
        elif r < 0.9: p = rnd.choice(PROSE).format(t=rnd.choice(TOPICS))
        else: p = rnd.choice(MISC)
        if p in seen: continue
        seen.add(p); rows.append({"id": len(rows), "prompt": p})
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} prompts -> {out}")

def post(port, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def gen(src, out, max_tokens=768, port=8080):
    done = set()
    try:
        for l in open(out): done.add(json.loads(l)["id"])
    except FileNotFoundError: pass
    t0, ntok = time.time(), 0
    with open(out, "a") as f:
        for l in open(src):
            r = json.loads(l)
            if r["id"] in done: continue
            j = post(port, {"messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": max_tokens, "temperature": 0})
            m = j["choices"][0]["message"]
            r.update(content=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
                     n=j.get("usage", {}).get("completion_tokens", 0))
            f.write(json.dumps(r) + "\n"); f.flush()
            ntok += r["n"]
            print(f"{r['id']}: {r['n']} tokens, {ntok / (time.time() - t0):.0f} tok/s overall", flush=True)

def replay(src, port=8080, start=0, end=1 << 30):
    for i, l in enumerate(open(src)):
        if i < start or i >= end: continue
        r = json.loads(l)
        msg = {"role": "assistant", "content": r["content"]}
        if r.get("reasoning"): msg["reasoning_content"] = r["reasoning"]
        post(port, {"messages": [{"role": "user", "content": r["prompt"]}, msg], "max_tokens": 1, "temperature": 0})
        if i % 50 == 0: print(f"replayed {i}", flush=True)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prompts": prompts(a[1], int(a[2]) if len(a) > 2 else 3000)
    elif a[0] == "gen": gen(a[1], a[2], *(int(x) for x in a[3:5]))
    elif a[0] == "replay": replay(a[1], *(int(x) for x in a[2:5]))
