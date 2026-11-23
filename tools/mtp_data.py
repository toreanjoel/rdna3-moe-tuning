#!/usr/bin/env python3
# mtp_data.py — MTP head training data: varied prompts, the model's own greedy answers, then a replay through a server started with STRATA_MTP_DUMP (final residual rows + tokens)
# usage: mtp_data.py prompts OUT.jsonl [N]          deterministic varied prompts (code-heavy, prose, Q&A, lists)
#        mtp_data.py selfinstruct OUT.jsonl [N] [port]    prompts written by the model itself across ~40 domains (resumable)
#        mtp_data.py gen PROMPTS.jsonl OUT.jsonl [max_tokens] [port]   the model's answers (resumable: skips done ids); rows hold "prompt" or a whole "messages" list (tools/sessions.py)
#        mtp_data.py replay GEN.jsonl [port] [start] [end]             [user, assistant(reasoning+content)] -> 1 token, so the
#                                                     prompt path dumps every row (start the server with STRATA_MTP_DUMP=<file>
#                                                     and --conversation-cache-mib 0)
import json, os, random, sys, time, urllib.request, zlib

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

DOMAINS = ["Python programming", "JavaScript and web frontends", "backend services and APIs", "databases and SQL",
           "DevOps, Linux and shell", "systems programming in C, C++ or Rust", "data science and statistics",
           "machine learning engineering", "mathematics homework", "physics and engineering", "biology and medicine",
           "history", "philosophy and ethics", "law and contracts (general information)", "personal finance",
           "business strategy and marketing", "writing and editing help", "creative fiction and poetry",
           "emails and workplace communication", "travel planning", "cooking and recipes", "fitness and health habits",
           "parenting and relationships", "language learning and translation", "product and UX design",
           "career advice and interviews", "debugging an error message", "explaining a concept simply",
           "comparing tools or technologies", "trivia and general knowledge", "games and puzzles", "music and film",
           "home repair and DIY", "science for kids", "security and privacy", "spreadsheets and Excel formulas",
           "mobile app development", "Chinese-language questions (write the requests in Chinese)",
           "agentic coding tasks with a file or stack trace pasted in", "long multi-part technical questions"]

def selfinstruct(out, n, port=8080, per_call=8):
    """Diverse prompts written by the model itself (sampled at T=1), deduplicated: closer to real chats than templates."""
    rnd = random.Random(4321)
    seen, rows = set(), []
    try:
        for l in open(out): r = json.loads(l); seen.add(r["prompt"]); rows.append(r)
    except FileNotFoundError: pass
    with open(out, "a") as f:
        while len(rows) < n:
            d = rnd.choice(DOMAINS); style = rnd.choice(["short and casual", "detailed with context", "a mix of lengths",
                                                         "phrased as a beginner", "phrased as an expert"])
            ask = (f"Write {per_call} different requests a real user might send to an AI assistant about {d}. Make them {style}, "
                   "varied in task and wording, each self-contained (include any code or data the request refers to). "
                   "Output only the requests, one per line, no numbering.")
            try:
                j = post(port, {"messages": [{"role": "user", "content": ask}], "max_tokens": 1200, "temperature": 1.0,
                                "top_p": 0.95, "seed": rnd.randrange(1 << 30)})
            except Exception as e:
                print("request failed:", e, flush=True); time.sleep(5); continue
            for line in (j["choices"][0]["message"].get("content") or "").splitlines():
                p = line.strip().lstrip("-*0123456789. )").strip().strip('"')
                if len(p) < 15 or p in seen: continue
                seen.add(p); r = {"id": len(rows), "prompt": p, "domain": d}; rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"{len(rows)} prompts", flush=True)
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
            # varied like real use (MTP_MIX=0: all greedy with thinking, as round 1): a third without thinking, half sampled at 0.6
            k = zlib.crc32(str(r["id"]).encode()) if os.environ.get("MTP_MIX", "1") != "0" else 0
            body = {"messages": r.get("messages") or [{"role": "user", "content": r["prompt"]}], "max_tokens": max_tokens,
                    "temperature": 0.6 if k % 2 else 0, "seed": k % 100000}
            if k % 3 == 1: body["chat_template_kwargs"] = {"enable_thinking": False}
            try: j = post(port, body)
            except Exception as e:
                print(f"{r['id']}: failed {str(e)[:100]}", flush=True); continue
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
        if not (r["content"] or r.get("reasoning")): continue
        try: post(port, {"messages": (r.get("messages") or [{"role": "user", "content": r["prompt"]}]) + [msg], "max_tokens": 1, "temperature": 0})
        except Exception as e: print(f"{r['id']}: replay failed {str(e)[:100]}", flush=True)
        if i % 50 == 0: print(f"replayed {i}", flush=True)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prompts": prompts(a[1], int(a[2]) if len(a) > 2 else 3000)
    elif a[0] == "selfinstruct": selfinstruct(a[1], int(a[2]) if len(a) > 2 else 3000, *(int(x) for x in a[3:4]))
    elif a[0] == "gen": gen(a[1], a[2], *(int(x) for x in a[3:5]))
    elif a[0] == "replay": replay(a[1], *(int(x) for x in a[2:5]))
