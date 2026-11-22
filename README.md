# rdna3-moe-tuning

Run **Qwen3.8-Flash-Next (IQ3_S, 82 GB MoE)** on a **single RX 7900 XTX (24 GB)** at **~111 tok/s**,
within a 294 W power cap. One script installs, tunes, serves and benchmarks
[Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3), a gfx1100-tuned fork of the Strata engine.

The model is 3–4× larger than VRAM. The engine keeps the most-used experts in VRAM (~90% hit rate), computes
the rest on the CPU in parallel, and drafts 4 tokens ahead with an MTP head.

## Summary

| | daily (`serve`) | perf mode (`perf`, opt-in) |
|---|---:|---:|
| generation, 8K / 32K context | 111.0 / 109.6 tok/s | 121.6 / 122.5 tok/s |
| clocks | core 2300, VRAM 1300 | core 3000, VRAM 1400 |
| power | 294 W cap, ~230 W used | 294 W cap, ~287 W used (coil whine) |

```bash
./strata-rdna3.sh install   # once: deps, engine + kernel patches, model, expert pack
./strata-rdna3.sh serve     # daily clocks, server on 127.0.0.1:8080 (HOST=0.0.0.0 for the LAN)
./strata-rdna3.sh perf      # same server, perf clocks; run `serve` again to go back
./strata-rdna3.sh status    # is it up, GPU power and VRAM
./strata-rdna3.sh monitor   # live view (q leaves it running)
./strata-rdna3.sh stop      # stop it (always stop before bench / quality)
./strata-rdna3.sh bench     # 8K + 32K benchmark
./strata-rdna3.sh build     # rebuild the engine with patches/ after changing them
```

Clock changes need root: install the passwordless helper once (`tools/strata-oc`, steps in its header).
After a reboot the card sleeps until `serve` or `perf` wakes it. Running two benchmark or quality jobs
at once has crashed the box, and VRAM 1500 MHz crashes it too.

Everything below is detail: hardware, measurements, every setting and everything tried.

## The box

| part | what |
|---|---|
| GPU | AMD Radeon RX 7900 XTX 24 GB (gfx1100), 294 W cap, core ≤2300 MHz, VRAM 1300 MHz, −50 mV |
| CPU | AMD Ryzen 9 7945HX (16 Zen 4 cores, 2 CCDs, AVX-512) |
| RAM | 92 GB DDR5-5200, dual channel |
| OS / stack | Ubuntu 24.04, ROCm 7.2.4 |
| model | Qwen3.8-Flash-Next GSQ-RCO IQ3_S (78 GB on disk), 128K context, int8 KV |

## Results

Single user, distinct prompts, 256 generated tokens, median of 5 runs, greedy.

```mermaid
xychart-beta
    title "Generation speed on this box (tok/s)"
    x-axis ["llama.cpp Vulkan", "llama.cpp ROCm", "Strata (stock)", "Strata-rdna3", "+ FAST + kernels (default)"]
    y-axis "tok/s" 0 --> 120
    bar [13, 25, 80, 103, 111]
```

```mermaid
xychart-beta
    title "Prompt reading, 32K tokens (tok/s)"
    x-axis ["Strata (stock)", "Strata-rdna3 (fork config)", "Strata-rdna3 (this repo)"]
    y-axis "tok/s" 0 --> 4000
    bar [2000, 3266, 3553]
```

| | prompt 8K | prompt 32K | generation 8K / 32K |
|---|---:|---:|---:|
| **this repo, `FAST=1` + kernel patches (default)** | **2,860** | **3,556** | **111 / 111.5** |
| this repo, `FAST=1`, before the kernel patches | 2,860 | 3,555 | 109–111 / 109 |
| this repo, `FAST=0` (exact answers) | 2,840 | 3,553 | 101 / 103 |
| fork's shipped config | 2,641 | 3,266 | 103–104 |
| fork author (9950X3D2, DDR5-6000) | 2,436 | 3,541 | 114–119 |
| stock Strata 0.1.62 | ~1,900 | ~2,000 | 71–85 |
| llama.cpp ROCm `-ncmoe 32 -ub 2048 -fa 1` | 1,112 (2K) | — | 25 |
| llama.cpp Vulkan `-ncmoe 22` | 103 (512) | — | 13 |

Real chats are faster than the benchmark because a conversation's earlier prompt is reused: coding-tool sessions
on this box log 122–137 tok/s generation (`./strata-rdna3.sh speed`).

### Where a generation step's time goes

One step verifies the MTP drafts and keeps ~2.1 tokens in ~20 ms. The GPU is the limit: it waits on the CPU's
experts for only ~1% of the step, so faster CPU code or more RAM bandwidth would barely help. GPU time per step
(8K context, `STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`):

```mermaid
pie title GPU time per step (21.7 ms, profiled)
    "attention / linear-attention layers" : 7.6
    "experts in VRAM" : 4.1
    "hyper-connections" : 3.4
    "router + shared expert" : 2.8
    "copy + combine" : 2.0
    "output head" : 0.7
    "waits (CPU, copies, gaps)" : 1.1
```

Raw numbers: [`snapshot/2026-11-21/`](snapshot/2026-11-21/) (stock Strata, llama.cpp, first fork sweeps),
[`snapshot/2026-11-22/`](snapshot/2026-11-22/) (CPU kernel / routing sweep with quality check, engine-switch
sweep, profile, kernel benches and sweeps 7–11, `bench-final.log`).

## Install

Needs Ubuntu 24.04 (or similar), ROCm 7.x in `/opt/rocm`, your user in the `render` and `video` groups,
~140 GB of free disk, and 64 GB+ RAM.

```bash
git clone https://github.com/toreanjoel/rdna3-moe-tuning && cd rdna3-moe-tuning
./strata-rdna3.sh install    # deps (sudo), engine build for gfx1100, model download if missing, expert pack
./strata-rdna3.sh serve      # tune the card, start the server in the background
```

The server speaks OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) on `127.0.0.1:8080`:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":512}'
```

Coding tools work against it with tool calls and streaming: OpenAI chat (Cursor, Continue, Cline, Roo, Aider),
Anthropic messages (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`) and OpenAI Responses (Codex).
`python3 tools/api_smoke.py [url]` checks all three (9/9 pass, 5 runs in a row). The model thinks before it
answers: give tools a max-tokens of 512 or more. [`examples/shop-cart`](examples/shop-cart) is a small coding
task to try a tool against it.

## Commands

| command | does |
|---|---|
| `install` | one-time setup (idempotent) |
| `serve` | apply tuning, (re)start the server; survives logout |
| `perf` | `serve` with perf clocks (core 3000, VRAM 1400), opt-in; [perf mode](#perf-mode-not-default) |
| `stop` / `status` | stop it / port, process, GPU power and VRAM |
| `monitor` | attach a live view (speed, requests, GPU); `q` detaches, server keeps running |
| `speed` | follow each request's prompt and generation tok/s from the log |
| `quality` | ~1 h: FAST vs exact on this box: next-token agreement, KL, perplexity, HumanEval pass@1 |
| `bench [lengths]` | the fork's benchmark (default `8192,32768`; `BENCH_TEMP=0.6` for sampled answers) |
| `tune` | apply clocks and power only |
| `uninstall` | remove everything the script created |

Everything lives under one folder (`R=`, default `~/work/rdna3`): `engine/`, `data/` (model, expert pack,
MTP head), `logs/`, `bench/`.

## Settings

Defaults are the measured best within the card's limits. Override any of them per run, for example
`FAST=0 ./strata-rdna3.sh serve`.

| variable | default | notes |
|---|---|---|
| `FAST` | 1 | skip low-ranked missed experts: +7–10% generation, answers change slightly ([below](#what-was-tried)); `0` = exact |
| `CTX` | 131072 | max context; the fork ships 262144 (128K frees ~1% more expert slots) |
| `WATTS` | 294 | power cap; the ceiling for this card |
| `SCLK` | 2300 | core clock max (MHz); kept low for coil whine, see [perf mode](#perf-mode-not-default) |
| `MCLK` | 1300 | VRAM clock (MHz); 1400 is stable and faster, 1500 crashes the box |
| `VO` | -50 | core voltage offset (mV) |
| `EPP` / `BOOST` | performance / 1 | the CPU computes the experts not cached in VRAM |
| `TABLE` | hipBLASLt `100202` | matches ROCm 7.2.x: prompts +8% |
| `HOST` / `PORT` | 127.0.0.1 / 8080 | `HOST=0.0.0.0` for the LAN |
| `XENV` / `XARGS` | — | extra engine env / flags, e.g. `XARGS="--spec 3"` |
| `NAME` / `REPEATS` | default / 3 | bench run name and repeats |

### Perf mode (not default)

How far the card goes with the clocks uncapped, still at the 294 W cap and -50 mV. It runs at the
power limit with coil whine, so it stays opt-in:
`./strata-rdna3.sh perf` (same as `SCLK=3000 MCLK=1400 ./strata-rdna3.sh serve`). A plain `./strata-rdna3.sh serve` goes back to daily clocks.
Clock changes need root; `tools/strata-oc` is a passwordless helper (install steps in its header).

| SCLK / MCLK | 8K tok/s | 32K tok/s | busy clock / power / hotspot |
|---|---|---|---|
| 2300 / 1300 (daily) | 111.0 | 109.6 | 2244 MHz, 230 W, 62 °C |
| 2300 / 1400 | 113.7 | 112.7 | 2247 MHz, 244 W, 67 °C |
| 2300 / 1450 | 112.9 | 111.6 | 2238 MHz, 248 W, 56 °C |
| 2300 / 1500 | — | — | box hard-crashed on apply |
| 2700 / 1400 | 118.7 | 119.9 | 2441 MHz, 284 W, 66 °C |
| **3000 / 1400** | **121.6** | **122.5** | 2505 MHz, 287 W, 70 °C |

At 3000 the card hits the 294 W cap and holds ~2500 MHz, so higher SCLK settings add little; more speed
would need more power. Raw output: [`snapshot/2026-11-22/sweep19.txt`](snapshot/2026-11-22/sweep19.txt) (`tools/sweep19.sh`, REPEATS 3).

## What was tried

| change | effect | kept |
|---|---|---|
| hipBLASLt table matching ROCm 7.2.x | prompts +7.5% (8K), +8.8% (32K) | yes |
| `--adapt-swaps 48` / `64` | within noise | no |
| `STRATA_AUX_CPUS=auto` | none | no |
| `--pool-workers 14` | none | no |
| `--pcie-frac 0.15` | generation −5% | no |
| `--spec-min-p 0.3`, `--lookup-chain 3` | within noise | no |
| `STRATA_MTP_DRAFT_ALL`, `MTP_QFUSE/MULTICOPY/GR_R8` | within noise | no |
| `STRATA_MEGAKERNEL=1` | within noise | no |
| `STRATA_IQ_PREFETCH` 1024 / 4096, `STRATA_CPU_YMM=1` (CPU expert kernels) | within noise | no |
| `STRATA_ROUTE_TAIL_SKIP=7` (`FAST=1`, default) | generation 101→111 (8K), 103→109 (32K); top-1 agreement 92.6%, KL 0.050, reference perplexity 4.064→4.041 | yes |
| `--max-context 131072` | 9,328→9,444 experts in VRAM | yes |
| `STRATA_ROUTE_TAIL_SKIP=8` | 109 / 106; KL 0.047 | no (7 is faster, same cost) |
| `STRATA_ROUTE_RESIDENT=0.5` / `1.0`, `STRATA_ROUTE_PRIOR=0.5` | 105–110; KL 0.08–0.09, top-1 ~90% | no |
| `STRATA_ADAPT_LAG=2 ADAPT_NOWAIT=1 ADAPT_OVERLAP=0` | two A/B pairs: +3% at 8K, −1% at 32K | no (noise) |
| `STRATA_SPEC_COUPLED=1` (± `GUMBEL`), sampled at T=0.6 | 108/114 → 101–105 | no |
| gfx1151 decode set (`LANECELL`, `MMVF_ROWS`, `TSUM`, `LFUSE`, `GDN_SPLIT`, `PLE_BATCH`, `EXPERT_V2`) | −2–3% | no |
| `STRATA_HC_REQ8=1` | 114 / 107, prompts −5% | no |
| `STRATA_COPY_HIP_PRIO=1`, `--host-core sibling` | −3% / none | no |
| higher core clock (no SCLK cap) | ~+1 tok/s, coil whine | no |
| llama.cpp (Vulkan, ROCm) | 4–8× slower generation | no |
| `HIP_FORCE_DEV_KERNARG=1`, `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` | none | no |
| `STRATA_RDNA_PAIR=1` (shared expert's gate + up in one GEMV) | 108.8 → 109.4, same answers | yes |
| `STRATA_GR_TAIL=1` ([`patches/gr-tail.patch`](patches/gr-tail.patch): hyper-connection read's serial tail) | 0.3–2.3 µs less per read; with PAIR 108.8 → 109.9, same answers | yes |
| `STRATA_EXP_AUTO=1` ([`patches/exp-auto.patch`](patches/exp-auto.patch): IQ2_S gate/up fused, Q2_0 down in LDS) | 109.7 → 110.7, same answers | yes |
| `STRATA_LFUSE=1`; hc read with the up weights preloaded | none | no |
| Q2_0 expert down on EXPDEC's staged kernel, 2 rows a lane (in `exp-auto.patch`; `STRATA_EXP_AUTO=5` = the LDS kernel) | kernel −10–23%; 110.6 → 110.8, same answers | yes |
| IQ4_XS GEMV at 1–2 tokens unrolled 2 ([`patches/kq-unroll.patch`](patches/kq-unroll.patch); `STRATA_RDNA_KQ_UNR=0` = stock) | kernel 56 → 26 µs; 110.6 → 110.8 (8K), 111.0 → 111.35 (32K), same answers | yes |
| RDNA Q4_K GEMV from 3 tokens, 2 on 6144-wide rows (in `kq-unroll.patch`; `STRATA_RDNA_KQ_MIN=0` = stock) | kernel −15–25% on 2–3 token windows; 110.9 → 111.25 (8K), 111.5 → 112.0 (32K); not bitwise (float order), vs stock top-1 100%, KL 0.00001 over 5K tokens | yes |
| `STRATA_EXPDEC=1` gate/up (the RDNA4 default) | IQ3_XXS gate/up +7% slower on gfx1100 | no |
| `--spec 3` / `5` / `6`, `--spec-min-p 0.7` | 108 / 112 / 114.5 / 112 (8K, 3 runs) | no |
| `--spec-min-p 0.35`, `--mtp-q4 proj` (full-precision draft head) | bench 8K +6% / +5%, 32K −2% / −0.4%; varied chats ([`tools/chatbench.py`](tools/chatbench.py)) 126.0 → 123.0 / 124.6 | no |
| DFlash2 drafter (`--draft dflash2`, incoai/Qwen3.8-Flash-Next-DFlash2) | CLI only in this build; 2.5–2.7 vs 2.2 tokens a round but 85–97 vs 100 tok/s (wider windows cost more), 0.9 GB VRAM | no |

The kernel changes are bit-identical to the engine's own kernels (checked on dumps and on the expert
bench's real rows), except the wider RDNA Q4_K GEMV, which sums in a different order: teacher-forced against
stock it keeps every top-1 token (KL 0.00001, `tools/quality18.sh`). Measured kernel facts behind them: a decode graph node costs ~2.8 µs (~3.2 ms of a 20 ms
step); the hyper-connection read is DRAM-bound for its first ~8 µs and then a serial tail; the expert layouts'
speed depends on the format and on entries per expert (`tools/expbench.sh`). rocprofv3 kernel times are not
usable inside HIP graphs on ROCm 7.2.4 (graph kernels share end timestamps).

Drafting is tuned out on this model: the MTP head keeps ~2.2 tokens a step and 37% of steps accept no draft, and
every setting that drafts more costs more verify time than it keeps on real chats. ~180 tok/s would need ~3.4 tokens
a step at today's step time, or a much shorter step.

Next: the GPU side of a step ([profile](#where-a-generation-steps-time-goes)), mainly fewer graph nodes (~3 ms of a
~20 ms step). Left out on purpose: higher clocks or
power, and 47 GB of reserved huge pages (transparent huge pages already back ~52 GB of the expert arena).

## Repo

| path | what |
|---|---|
| `strata-rdna3.sh` | the script |
| `tools/` | benchmark sweeps, quality check (`quality.sh`, `humaneval.py`), API smoke test, varied-chat speed (`chatbench.py`), CLI generation with any engine flags (`dfcli.py`, `mkprompt.py`) |
| `examples/shop-cart/` | a small coding task for trying a tool against the server |
| `patches/` | engine kernel patches, applied on top of the pinned fork by `./strata-rdna3.sh build`, each behind its own switch |
| `snapshot/<date>/` | raw measurements behind every number here |
| [`docs/history-stock-strata.md`](docs/history-stock-strata.md) | the stock-Strata investigation (clocks, power, decode ladder) |
| [`docs/kernel-work-notes.md`](docs/kernel-work-notes.md) | running kernel-work log: measurements, decisions, what was dropped and why, box quirks |
| `tools/strata-oc` | optional root helper for clocks/power (install steps in its header; caps power at 294 W) |
