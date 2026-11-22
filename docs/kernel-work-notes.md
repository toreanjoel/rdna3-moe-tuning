# Handoff: GPU kernel work for decode (start here after /clear)

State on 2026-11-22 (pushed, origin/main ac9c9d1): FAST=1 default, 128K ctx, ~109-111 tok/s greedy bench (8K/32K),
real chats 122-137. Box: see README "The box". Access: `.nullrouter/tools/rt.sh '<cmd>'` (key SSH). Live server on
127.0.0.1:8080 (reverse tunnel, keep the port). Engine source on the box: ~/work/rdna3/engine (fork nicejunjie/Strata-rdna3
at PIN in strata-rdna3.sh, built into engine/engine/strata, build dir build-hip).

## Where the time goes (snapshot/2026-11-22/sweep6-profile.txt)
Step ~20 ms, ~2.1 tokens kept, avg window 2.8 tokens. GPU-bound: GPU waits on CPU experts ~0.2 ms/step (~1%).
GPU ms/step (profiled 21.7): attention/GDN layers 7.6, VRAM experts 4.1, hyper-connections (hc0/hc1 norm/down/up) 3.4,
router+plan+bell + shared+quant 2.8, copy+combine 2.0 (1.5 at 8K, 0.8 at 32K), head 0.7, waits 1.1. Draft (MTP) 1.7 ms on top.
Tools: XENV="STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1" NAME=x REPEATS=1 ./strata-rdna3.sh bench; grep "decode GPU stages" bench/x.

## Rules for this work
- All work lives in THIS repo: engine changes as patch files (e.g. patches/*.patch) that strata-rdna3.sh applies at
  install/patch time on top of PIN, plus an env/flag to turn each off. No second project, no fork repo to manage.
- Each change: exact (bit-identical) preferred; verify with quant_quality.py (KL vs ref, see tools/quality4.sh) and the
  bench (REPEATS=5, interleaved A/B pairs, noise ~±3 tok/s). Keep raw numbers in snapshot/<date>/.
- Power/clock limits fixed (memory box-power-limits). One model. Port 8080 back up after every sweep.
- At the end: README charts/tables for this box updated, then push only after the user confirms.

## Measured 2026-11-22 (snapshot/2026-11-22/kprof-k3.txt, sweep7.txt, sweep8.txt, kbench-*.txt)
- rocprofv3 per-kernel TIMES are invalid inside HIP graph replays on ROCm 7.2.4 (graph kernels share end timestamps;
  proven with an 8-kernel spin graph). Names/counts are valid. tools/kprof.sh preloads the tool into the engine only
  (via XENV) and stops via server QUIT (tools/kprof-stop.sh); a signal to the engine or rocprofv3 around the launcher loses the trace.
- Decode graph: 48 layers (36 GDN + 12 QSA), ~1140 nodes per 4-token window (~24/layer) + MTP ~120.
- Node cost (STRATA_DUMMY_NODES, sweep7): ~2.8 us per empty node per step -> ~3.2 ms of a ~20 ms step is node overhead.
  HIP_FORCE_DEV_KERNARG=1 and DEBUG_CLR_GRAPH_PACKET_CAPTURE=0: no change (sweep8).
- Isolated kernels (claude_bench, production switches, includes ~2.8 us node cost): hc read int8 (rdna_gr q8) T1 14.2 us,
  T2 17.7, T3 20.3, T4 23.3 for ~7 MB (300-490 GB/s); 96 reads/step ~2 ms. Head Q6_K T1 568 us (917 GB/s, at roofline).
  IQ3_S experts 2560x6144 ~17-26 us (250-400 GB/s); shexp 2560->640 ~6 us (latency-bound).
- Per-layer small nodes worth folding into neighbours: native_swiglu_quantize_q8_1 (48), swiglu_q8_1_entries (48),
  wait_flag_copy_i32 (48), route_publish_res (48), combine_cpu_k10 (48), apply/norm (36 each).
- Build: `./strata-rdna3.sh build` (patches/*.patch, reconfigures the moved build-hip; TESTS=ON TARGETS="claude_bench
  fused_gr_bench" for benches; tools/kbench.sh runs them with the server stopped).

## hc read deep-dive 2026-11-22 (patches/gr-tail.patch, STRATA_GR_TAIL=1, STRATA_GR_TRACE=1 for phase stamps)
- In-kernel wall-clock stamps (claude_bench gr): the load phase is DRAM-bound (~8 us for 6.7 MB at T1, ~840 GB/s);
  the rest is a serial tail growing ~1 us/token: staging (L2 re-reads by all 41 row groups), barrier wait, up fold+dot.
- Preloading up weights before the down pass: no gain (dropped). Hoisting epilogue loads: no gain (dropped).
- GR_TAIL (token-by-token staging with sumsq folded in, unrolled inject rows, one-shot partial loads): bit-identical
  (dumps T1-6), -0.3/-1.5/-0.6/-2.2/-1.5/-2.3 us at T1-6. End to end at 8K: inside noise (sweep9).
- sweep10: RDNA_PAIR +0.6, PAIR+GR_TAIL +1.0 tok/s (109.9 vs 108.8, both rounds); LFUSE nothing. Both now default
  (KERNENV in strata-rdna3.sh), pushed 5e52ffd. Next fold idea: SwiGLU+q8_1 into the pair GEMV's tail (q8_tail pattern).

## Candidate targets (original guesses; see measurements above)
1. Hyper-connections 3.4 ms: many tiny norm/down/up kernels per layer -> fuse into the neighbouring stage.
2. copy+combine 0.8-1.5 ms + router/shared 2.8 ms: fuse router top-k, shared expert and combine.
3. VRAM expert GEMV 4.1 ms (IQ3_S on gfx1100, window 2-5 tokens): check achieved bandwidth vs ~960 GB/s; tune tiles.
4. Draft 1.7 ms: MTP head cost; and raise tokens kept per step (2.1 of 2.8).
First step: per-kernel timing (rocprofv3 --kernel-trace on a short bench) to see launch gaps vs bandwidth.

## Expert down + drafting 2026-11-22 (commit 0db9bc1 local, not pushed; snapshot/2026-11-22/sweep12.txt, expbench-expdec.txt)
- EXPDEC's staged Q2_0 down kernel (expdec_down_r<42,2>, auto rows/block) now taken under STRATA_EXP_AUTO=1 (=5: old mode 8 LDS
  down). Isolated -10..-23% vs mode 8, bit-identical; end to end 110.6 -> 110.8 (2 rounds, REP 5). EXPDEC gate/up: slower on gfx1100.
- Path to ~180 tok/s is tokens per window, not kernels: base keeps ~2.2 tokens/window at ~19.6 ms (accepted 0.63 of drafts).
- sweep13 (8K, REP 3, decode tok/s): base 110.6, --spec 3 108.1, 5 112.0, 6 114.5, --spec-min-p 0.35 118.2, 0.7 112.2,
  spec 6 + min-p 0.35 98.6, --mtp-q4 proj (full-precision draft head; accept 0.63 -> 0.69) 116.8.
- DFlash2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2 @a22a1744, hashes match strata_control/drafting.py) packed to
  ~/work/rdna3/data/dflash2/qwen3.8-flash-next-dflash2.q8.pack (tools/dflash2_pack.py). Flags: --draft dflash2 --draft-model PACK
  [--draft-cap N --draft-conf P]. sweep13 SET=df tests it.
- sweep13 SET=df: --draft dflash2 is CLI-only in this build ("off ... not --serve"), the server falls back to MTP. Accept counts in
  serve runs are deterministic (s13df == s13base exactly), so tok/s spread is timing noise only.
- CLI (tools/dfcli.py + tools/mkprompt.py, bench 8K prompt r0, 256 tokens): MTP 2.21 tok/round 100.3 tok/s; min-p 0.35 2.27, 99.8;
  DFlash2 cap 6: 2.47, 97.4; cap 8: 2.49, 84.8; cap 8 conf 0.3: 2.71, 90.7. Drafting ~2.86 ms/round either way. 37% of rounds accept
  0 drafts with MTP. Verdict: DFlash2 not worth it here (wider windows cost more than they keep; no --serve).
- sweep14: REP 5 x 2 rounds, 8K+32K, base vs --spec-min-p 0.35 vs --mtp-q4 proj.
- sweep14 (REP 5 x2): 8K base 110.7, min-p 0.35 117.2, mtp-q4 proj 116.6; 32K base 111.4, min-p 0.35 109.4, q4 proj 110.9.
- sweep15 (tools/chatbench.py, 6 varied chat prompts x2, live server): base 126.0, q4 proj 124.6, min-p 0.35 123.0. The bench-prompt
  gains do not carry to real chats -> drafting defaults stay (--spec 4 --spec-min-p 0.5 --mtp-q4 all). Drafting is a dead end here;
  the remaining lever is GPU step time (graph nodes ~3 ms/step, attention 7.6 ms, experts 4 ms).

## Dense GEMVs at the model's real shapes 2026-11-22 (patches/bench-shapes.patch adds them to claude_bench mmvq; tools/gguf_tensors.py)
- Per layer: GDN qkv Q6_K 2560->10240, z (attn_gate) Q4_K 2560->6144, ssm_out Q6_K 6144->2560, alpha/beta bf16 2560->48;
  QSA q IQ4_XS 2560->12288, k/v Q6_K 2560->512, output Q4_K 6144->2560; shexp gate/up Q4_K / IQ4_XS 2560->640, down IQ4_NL.
- claude_bench (T1-4): Q6_K qkv 26-30 us (720-816 GB/s, fine); Q4_K z/out 19-32 us (270-460 GB/s, latency-bound, 1440 B rows).
- IQ4_XS cliff: rdna_kq UNR 5 (stock at 1-2 cols) -> T2 56.7 us vs 25.5 at UNR 2 (4/5 cliff at other widths). Fix: UNR 2
  default for IQ4_XS (patches/kq-unroll.patch, STRATA_RDNA_KQ_UNR=0 stock). Bitwise. sweep16: 110.6->110.8 (8K), 111.0->111.35 (32K).
- Tried, dropped: more unroll for Q4_K (no gain); rdna_kq with 2/4 rows per wave sharing activations (bitwise, but slower: fewer waves).
- rdna Q4_K is bitwise the wave kernel (rel 0) and faster from 3 cols (2 at n_in 6144): STRATA_RDNA_KQ_MIN (default on, same patch).
  claude_bench: z T3 31.4->26.6, out T2 26.0->21.0, T3 29.4->22.8, shexp T3 11.7->10.2. sweep17 = end to end.
- Box note: ~/work/rdna3 has no patches/ dir; the engine source there carries the edits directly. Verify the repo patches on a clean
  PIN checkout before pushing.
- sweep17: KQ_MIN 110.9->111.25 (8K), 111.5->112.0 (32K), both rounds. Kept.
- CORRECTION: KQ_MIN is NOT bitwise end to end (outputs differ). quality18 (KQ_MIN=1 vs 0, teacher-forced 5K tokens):
  top-1 100%, KL 0.00001, PPL 4.040 vs 4.041. Kept; README says "not bitwise".
- Branchless-scale Q4_K (sweep18, kbench18): 10-30% faster isolated, -0.5 tok/s end to end, not bitwise. Reverted.
- Box crashed twice (overlapping quality runs, then 15:54 on 2026-11-22). After a reboot the card is runtime-suspended:
  sysfs OD reads return EBUSY until root writes power/control=on. tools/strata-oc (root helper, NOPASSWD sudoers,
  installed on the box 2026-11-22) does the wake + clocks; tune() uses it when present.
- OD ranges (7900 XTX): SCLK 500-5000, MCLK 97-1500, VDDGFX offset -450..0, power cap max 402 W (we stay <=294).
  Perf-mode test = tools/sweep19.sh (not a default).
- sweep19 (2026-11-22, 294 W, VO -50, REP 3), 8K/32K tok/s: 2300/1300 111.0/109.6; 2300/1400 113.7/112.7;
  2300/1450 112.9/111.6; 2300/1500 HARD CRASH on apply; 2700/1400 118.7/119.9; 3000/1400 121.6/122.5
  (power-limited at ~287 W, ~2505 MHz busy). Perf mode = SCLK=3000 MCLK=1400, opt-in only (user: whine, not default).
  Daily MCLK could move to 1400 (+2.7 tok/s, stable here) but the user hasn't decided; defaults unchanged.
- More power (up to the card's 402 W) would likely give ~126-130 tok/s in perf mode; the user chose to bank 294 W.
  `./strata-rdna3.sh perf` = SCLK 3000 / MCLK 1400. sweep19 takes POINTS="sclk:mclk:watts" if this is revisited
  (strata-oc must then allow >294 and be reinstalled as root).
- Strata 0.2.19 (upstream Niko1221/Strata, 2026-11-22) is NOT what we run: we pin nicejunjie/Strata-rdna3 6c2cd28b.
  Tested 2026-11-22 (tools/upbench.sh, shadow ROOT, bundle in ~/work/strata/0.1.65, 1.1 GB): upstream defaults
  90.1/86.9 tok/s, prompt 2603/3419 vs ours 109.2/108.5, 2860/3554. Fork env (STRATA_GR_RDNA etc.) segfaults it in
  fused_gr_read_multi capture. Its RDNA3 notes: IQ2_XS/NVFP4 LDS decode (em0=8), PF_F16=2 (fork already sets it),
  moe_p1a FL=3 (prefill). Nothing worth porting for our IQ3_S expert formats. Kept ours.
