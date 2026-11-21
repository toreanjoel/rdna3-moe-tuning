#!/usr/bin/env python3
# api_smoke.py — checks the endpoints coding tools use (OpenAI chat, Anthropic messages, OpenAI Responses), with tools and streaming.
# usage: python3 tools/api_smoke.py [base_url]   (default http://127.0.0.1:8080)
import json, sys, time, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
WEATHER_OA = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
WEATHER_AN = {"name": "get_weather", "description": "Current weather for a city",
              "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
WEATHER_RS = {"type": "function", "name": "get_weather", "description": "Current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
ASK = "What's the weather in Paris right now? Use the tool."
results = []


def post(path, body, stream=False, headers=None):
    h = {"Content-Type": "application/json", "Authorization": "Bearer x", "x-api-key": "x",
         "anthropic-version": "2023-06-01", **(headers or {})}
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), h)
    r = urllib.request.urlopen(req, timeout=300)
    if not stream:
        return json.loads(r.read())
    events = []
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            d = line[5:].strip()
            if d and d != "[DONE]":
                events.append(json.loads(d))
    return events


def check(name, fn):
    t = time.time()
    try:
        ok, note = fn()
    except Exception as e:  # noqa: BLE001
        ok, note = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, note, time.time() - t))
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {time.time() - t:5.1f}s  {note}", flush=True)


def models():
    d = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=30).read())
    ids = [m["id"] for m in d.get("data", [])]
    return bool(ids), ",".join(ids)


def oa_plain_stream():
    ev = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi in 3 words."}],
                                       "max_tokens": 512, "stream": True}, stream=True)
    ch = [c for e in ev for c in e.get("choices", [])]
    txt = "".join((c["delta"].get("content") or "") for c in ch)
    rsn = "".join((c["delta"].get("reasoning_content") or c["delta"].get("reasoning") or "") for c in ch)
    fin = next((c.get("finish_reason") for c in reversed(ch) if c.get("finish_reason")), None)
    return bool(txt.strip()), f"{txt[:60]!r} finish={fin} reasoning={rsn[:80]!r}"


def oa_tools(stream):
    def f():
        body = {"messages": [{"role": "user", "content": ASK}], "tools": [WEATHER_OA], "max_tokens": 512, "stream": stream}
        if not stream:
            m = post("/v1/chat/completions", body)["choices"][0]["message"]
            tc = m.get("tool_calls") or []
            args = tc[0]["function"]["arguments"] if tc else ""
        else:
            ev = post("/v1/chat/completions", body, stream=True)
            tc, args = [], ""
            for e in ev:
                for c in e.get("choices", []):
                    for t in c["delta"].get("tool_calls") or []:
                        tc.append(t)
                        args += (t.get("function") or {}).get("arguments") or ""
        ok = bool(tc) and "paris" in json.loads(args or "{}").get("city", "").lower()
        return ok, f"{len(tc)} tool_call chunk(s), args={args[:60]!r}"
    return f


def oa_tool_roundtrip():
    msgs = [{"role": "user", "content": ASK},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 17, \"sky\": \"cloudy\"}"}]
    m = post("/v1/chat/completions", {"messages": msgs, "tools": [WEATHER_OA], "max_tokens": 512})["choices"][0]["message"]
    txt = m.get("content") or ""
    return "17" in txt, repr(txt[:80])


def an_tools(stream):
    def f():
        body = {"model": "local", "max_tokens": 512, "tools": [WEATHER_AN], "messages": [{"role": "user", "content": ASK}],
                "stream": stream}
        if not stream:
            d = post("/v1/messages", body)
            tu = [b for b in d.get("content", []) if b.get("type") == "tool_use"]
            inp = tu[0]["input"] if tu else {}
            stop = d.get("stop_reason")
        else:
            ev = post("/v1/messages", body, stream=True)
            tu = [e for e in ev if e.get("type") == "content_block_start" and e["content_block"].get("type") == "tool_use"]
            js = "".join(e["delta"].get("partial_json", "") for e in ev
                         if e.get("type") == "content_block_delta" and e["delta"].get("type") == "input_json_delta")
            inp = json.loads(js or "{}")
            stop = next((e["delta"].get("stop_reason") for e in ev if e.get("type") == "message_delta"), None)
        ok = bool(tu) and "paris" in str(inp.get("city", "")).lower() and stop == "tool_use"
        return ok, f"tool_use={len(tu)} input={inp} stop={stop}"
    return f


def an_count():
    d = post("/v1/messages/count_tokens", {"model": "local", "messages": [{"role": "user", "content": "hello there"}]})
    return d.get("input_tokens", 0) > 0, str(d)


def rs_tools():
    d = post("/v1/responses", {"input": ASK, "tools": [WEATHER_RS], "max_output_tokens": 512})
    fc = [o for o in d.get("output", []) if o.get("type") == "function_call"]
    args = fc[0].get("arguments", "") if fc else ""
    return bool(fc) and "paris" in args.lower(), f"function_call={len(fc)} args={args[:60]!r}"


check("GET /v1/models", models)
check("openai chat stream (text)", oa_plain_stream)
check("openai chat tools", oa_tools(False))
check("openai chat tools stream", oa_tools(True))
check("openai tool result round-trip", oa_tool_roundtrip)
check("anthropic messages tools", an_tools(False))
check("anthropic messages tools stream", an_tools(True))
check("anthropic count_tokens", an_count)
check("openai responses tools", rs_tools)
bad = [r for r in results if not r[1]]
print(f"\n{len(results) - len(bad)}/{len(results)} passed")
sys.exit(1 if bad else 0)
