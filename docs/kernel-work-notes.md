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

## Roofline and the 200 tok/s question 2026-11-22 (tools/roofline.py, snapshot/2026-11-22/roofline.txt)
- Bytes a step must read (GGUF types, 2.8-token window, 90% expert hit, int8 HC, 8K): ~6.3 GB -> floor ~7.7 ms at
  816 GB/s (85% of 960). Profiled now ~23.4 ms incl. draft (bench step ~19-20 ms). 2.1 tokens kept: floor 272 tok/s,
  floor + 3 ms other cost ~196. So ~200 needs every stage near DRAM roofline AND ~1100 graph nodes' overhead gone.
- Gap by stage (now - floor ms): GDN/attention 4.8 (303 GB/s), HC 2.6 (187), router+shared 2.5 (97), copy+combine 2.0,
  draft 1.55, waits 1.1, experts 1.1 (596 GB/s), head 0.06 (at roofline). CPU RAM bandwidth is not the limit (GPU waits ~1%).
- sweep20: STRATA_MEGAKERNEL=1 is a no-op here: the persistent GDN kernel (megakernel_gdn.cu, bitwise the six nodes) only
  covers Q8_0 projections; ours are Q6_K qkv / Q4_K z / Q6_K out ("types 14/12/14"). SIDE_MIXER=1 97.5, =2 82.3 vs 109.3 (worse).
- mk_gdn_parity at one card's geometry (patches/mk-parity-geom.patch: STRATA_MK_HK=16 STRATA_MK_HV=48; Q8_0 weights,
  STRATA_MK_REFSPLIT=1 splits per kernel; snapshot/2026-11-22/mk-gdn-parity.txt): six kernels 99.9 / 108.6 us a layer
  (T1 / T3), persistent 101.8 / 104.6 -> the megakernel saves ~0 on gfx1100 (graph node cost is small here). The GEMVs
  inside are already near roofline (qkv Q8_0 27.9 MB in 34 us = 820 GB/s). DROPPED: porting Q6_K/Q4_K into it.
- Real stage split (sweep6-profile, ms/window): the gaps are latency-bound small kernels, not big GEMVs:
  router+plan+bell 1.03 + 0.35 and shared+quant 1.05 + 0.35 (~29 us a layer for ~3.5 MB -> ~2.6 ms vs ~0.2 floor);
  hc down/up ~15 us each for 3.3 MB (3.4 ms vs 0.8); copy+combine 0.8-2.4; z Q4_K 0.96 (floor 0.39); QSA misc ~2.5.
- Honest ceiling: removing most latency gaps (horizontal fusion: router + shexp gate/up in one launch over the same
  input, hc folds, combine folds) is worth ~3-4 ms of ~20 -> ~130-135 tok/s daily (~145 perf mode). 200 also needs
  ~3 tokens kept a step; MTP keeps 2.1 and DFlash2 (2.5-2.7) costs more than it keeps (sweep13).
- Next: (1) router GEMV + shared gate/up as one multi-matrix launch (largest latency gap); (2) hc folds; (3) combine;
  each behind a switch, bitwise check, end-to-end A/B.

## Stage cost by doubling 2026-11-22 (tools/sweep21.sh, STRATA_ABL bitmask in verify.cpp; snapshot/2026-11-22/sweep21*.txt)
- Skipping a stage is useless: wrong outputs scatter routing, CPU experts explode (host 10-21 ms), T changes (sweep21a-c).
- Running a stage TWICE keeps outputs and windows identical; the added ms is the stage's real cost (a lower bound: the
  second pass can hit the 96 MB Infinity Cache). Base verify 17.9-18.6 ms/window (T 2.74-2.79):
  shared expert (&1) +0.58, VRAM experts (&2) +2.2, all mixer dense GEMVs (&4) +2.2, router GEMV (&8) +0.2.
- So the per-stage profile (sweep6) inflates small stages ~2x (stamp nodes). Dense GEMVs move ~1.9 GB/step in ~2.2 ms
  (~850 GB/s, at roofline). Shared expert + router are worth <=0.8 ms total -> fusing them is <=4%; deprioritised.
- ~13 ms of the window is neither dense GEMVs nor experts: HC reads, GDN conv/ab/rec, QSA attention/top-k, combine,
  plan/publish, waits, node overhead (~1140 nodes), draft 1.8.
- Caveat: the doubled pass can hit the Infinity Cache, so dense (~1.9 GB) and experts are under-counted; the profile's
  4.5 / 4.1 ms are closer to their real cost. Warm re-reads at ~half time = the cache is worth ~2 ms IF filled for free.
- sweep22 (--pool-workers 4/12/16): 12 = 16 (109.2 / 109.1), 4 = 100.3. The CPU experts (system RAM, ~35 GB/s of ~83)
  are not on the critical path at 12 workers; faster CPU experts would save <=0.25 ms (GPU waitCPU). Not a lever.

## Infinity Cache prefetch 2026-11-22 (DROPPED; tools/mallbench.hip, sweep23a-d, snapshot/2026-11-22/prefetch-abl-experiment.diff)
- mallbench (gfx1100): 32-64 MB streaming read cold 684-779 GB/s, warm 1.2-2.0 TB/s; 128 MB no longer fits (854).
  A 24-block side read pulls ~330 GB/s (48: 595). So prefetching next-layer weights during latency-bound phases looked worth ~2 ms.
- In-graph side branch (fork/join per layer, STRATA_PREFETCH=1/2/3): even a 2.6 MB router prefetch costs +3.2 ms/window
  (~65 us per fork/join on HIP graphs). DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 / DEBUG_HIP_FORCE_GRAPH_QUEUES=2: no change.
  This is also why STRATA_SH_STREAM / SIDE_MIXER lose here: graph branches are expensive on this stack.
- Out-of-graph persistent prefetcher (=4: own stream, polls the doorbell ring m_seq_, reads layer L+1 after layer L routes):
  24 blocks 101.2 tok/s, 48 blocks 55.0 (vs 109.5). Concurrent blocks delay kernels that need their whole grid resident
  (the fused HC read's inter-block barrier). Only an in-kernel prefetch (idle blocks of an existing launch) could still work.

## Expert kernels vs read speed 2026-11-22 (expbench-e24: STRATA_EXP_AUTO=1, 20 groups)
- Per layer: gate/up 43-55 us (380-560 GB/s), down 24-34 us (320-600 GB/s), ~75-90 us -> ~3.8 ms/window. A plain
  streaming read of the same ~25 MB takes ~42 us on gfx1100 (mallbench: 16 MB 36 us, 32 MB 49 us) -> the expert kernels
  are near the practical read rate for their size; the fixed per-kernel cost (~15-20 us) dominates, not decode ALU.
  Conclusion: fewer launches beats faster bodies everywhere on this engine.

## Mixed-format GEMV pair 2026-11-22 (patches/rdna-mix.patch, STRATA_RDNA_MIX default on, =0 off; sweep24, quality25)
- One launch for two matrices of different RDNA formats (12/13/14/23) over the same q8_1 input; rows of each run their own
  rdna_kq / rdna_q6k body (kernels refactored into __device__ bodies + thin wrappers). Used for GDN qkv (Q6_K) + z (Q4_K)
  and the shared expert's gate + up when the formats differ (most layers: Q4_K/IQ4_XS/Q5_K/Q6_K mixes).
- Bitwise vs the separate RDNA kernels (KQ_ALL=1 with/without MIX: identical windows/tokens). Verify -0.32 ms/window
  (18.53 -> 18.21, 17.83 -> 17.51). Bench tok/s moved 109.3 -> 108.8 only because answers changed (2.23 -> 2.10 tokens kept
  on these two prompts; KQ_ALL alone shows 117.8 the other way) - with 2 bench prompts, acceptance noise is +-8%: judge
  non-bitwise changes on verify ms/window, not tok/s. quality25: top-1 100%, KL 0.00000, PPL 4.041 vs 4.040.

## Fused router 2026-11-22 (patches/route-fuse.patch, STRATA_ROUTE_FUSE default on, =0 off; tools/sweep25.sh)
- Router GEMV (512 bf16 rows) + the shared gate's aux row + top-10 + doorbell publish in one launch: the same GEMV code
  per block, then the last block to finish (atomic ticket, fences) runs route_row and the publish tail. 48 launches/step fewer.
- Bitwise: answers identical (4/4 SAME), same windows. Verify 18.22 -> 18.20, 17.52 -> 17.49 ms/window (2 rounds each):
  ~0.6 us a layer. A removed launch inside the graph is worth well under 1 us here; the gaps are the dependent chain's
  per-kernel latency, not launch issue. More launch merging only pays where it also shortens the chain.

## IQ2_S gate/up pre-signed 2026-11-22 (patches/gu-n.patch, STRATA_GU_N default on, =0 off; sweep26, sweep27, expbench-gn*)
- STRATA_GU_SWIGLU=1 (the 512-thread gate/up + SwiGLU + q8_1 kernel, bitwise): +0.3 ms/window slower here (sweep26). Dropped.
- Production expert kernels (tools/knames.py on kprof-k4): IQ2_S gate/up = native_gu_fused_kernel<22> (20 layers), IQ3_XXS /
  IQ3_S = native_gu_multi_kernel + swiglu_q8_1_entries, IQ4_NL down = native_down_multi_kernel<20>, Q2_0 = expdec_down<42,2>.
- IQ2_S was the outlier: DotG<22,1> applies signs per token (12 dp4a); IQ3_* already pre-sign. dotg22_n signs the grid
  words once per row ((g ^ m) + (m & 0x01010101); iq2s_grid bytes are only 8/25/43, no carry) -> 8 dp4a a token. Exact
  integer sums -> bitwise (expbench "bitwise equal", answers SAME). Isolated gu at 20 groups: T1 43 = 43 us (path kept
  for 1-token passes: the pre-signed one was +1.5 us there), T2 55 -> 48-49, T3 73 -> 61. End to end -0.03 ms/window
  (18.23/18.22 -> 18.19/18.21). 4 rows a warp pass (more loads in flight): no gain, dropped.

## CPU experts, plan, draft switches 2026-11-22 (sweep28-30, tools/membw.c, claude_cpubench, tools/knames.py)
- Fresh profile (sweep28, stamps inflate small stages): copy+combine 1.76 GDN + 0.59 QSA ms/window (1.5-3.1 run to run) is
  mostly the GPU spinning in combine_cpu for the CPU experts (~2.0 experts / 2.6 entries a layer-window, 92% VRAM hits,
  9444 slots, VRAM full: 360 MiB free).
- CPU side is at the RAM floor: membw (AVX-512, cores 1..N) streams 48-53 GB/s, a cold 4 MB slice takes ~93 us at 12-15
  threads. claude_cpubench (real rows, 12 workers): 1 expert 60 us (36 GB/s), 2 experts 101 us cold / 77 us L3-warm, 6
  experts ~50 GB/s. Warm is only 1.1-1.35x -> L3 prefetch of predicted misses is worth <=25% of the CPU path; not built.
  --pcie-frac (GPU fetches misses over PCIe) was -5% earlier. The CPU wait is structural at this hit rate.
- STRATA_VERIFY_DEVICE_PLAN=1 (sweep29): +1 ms/window and answers changed here. Dropped.
- sweep30: STRATA_MTP_QFUSE=1 / MULTICOPY=1 noise (bitwise); STRATA_MTP_HIP_PRIO=1 draft 1.72 -> 2.7-2.9 ms (worse). None kept.
- Kernel count per layer-window ~25 (knames on kprof-k4); removing one dependent launch is worth ~0.5-1 us here (route fuse).
- IQ4_NL down with one unaligned 8-byte weight load instead of four 16-bit loads (bitwise): 30.7 / 33.9 us vs 30.8 / 33.4
  (G 20, T 1 / 2) -> no gain, reverted. Marginal rates from G 10 -> 20: down ~710 GB/s, gate/up ~680 GB/s; each expert
  kernel carries a ~5-10 us fixed ramp/tail. Body tuning is done; only fewer dependent kernels or more overlap would pay.

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
  Daily MCLK moved to 1400 on 2026-11-23 (user approved; +2.5%, stable). SCLK 2300 / 294 W unchanged.
  bench d1400 (2026-11-23, all patches, REP 3): 8K 117.7 / 32K 117.4 decode, prefill 2903 / 3605 (was 113-114 / 114 at MCLK 1300).
- More power (up to the card's 402 W) would likely give ~126-130 tok/s in perf mode; the user chose to bank 294 W.
  `./strata-rdna3.sh perf` = SCLK 3000 / MCLK 1400. sweep19 takes POINTS="sclk:mclk:watts" if this is revisited
  (strata-oc must then allow >294 and be reinstalled as root).
- Strata 0.2.19 (upstream Niko1221/Strata, 2026-11-22) is NOT what we run: we pin nicejunjie/Strata-rdna3 6c2cd28b.
  Tested 2026-11-22 (tools/upbench.sh, shadow ROOT, bundle in ~/work/strata/0.1.65, 1.1 GB): upstream defaults
  90.1/86.9 tok/s, prompt 2603/3419 vs ours 109.2/108.5, 2860/3554. Fork env (STRATA_GR_RDNA etc.) segfaults it in
  fused_gr_read_multi capture. Its RDNA3 notes: IQ2_XS/NVFP4 LDS decode (em0=8), PF_F16=2 (fork already sets it),
  moe_p1a FL=3 (prefill). Nothing worth porting for our IQ3_S expert formats. Kept ours.
  Bundle deleted afterwards (~/work/strata removed). To retest a future release: download its hip-rx7900 tarball to
  ~/work/strata/<ver>, then `UP=~/work/strata/<ver> bash tools/upbench.sh` (stops/restarts the server, our engine untouched).

- 2026-11-23 IQ3_XXS (ISTA GSQ-RCO, tools/sweep31.sh, daily tune, REP 3, interleaved S/XXS/S/XXS). Shard 2 is
  byte-identical to IQ3_S's (hard link, only 47 GB new); pack data/packs/iq3_xxs 42 GB; setup iq3xxs-1gpu.
  Decode 8K 113.5 vs 114.5, 32K 119.1 vs 114.3; prompt 8K 3043 vs 2866. Expert slots 9444 -> 11252, CPU experts per
  layer-window 0.99 -> 0.37, host per layer 2.25 -> 0.89 ms, but GPU-reach wait 13.8 -> 14.8 ms (IQ2_XXS/IQ2_XS
  expert kernels untuned), net ms/window 18.7 -> 18.4. Quality vs IQ3_S: top-1 85.0%, KL 0.184, PPL 4.022 vs 4.041;
  ISTA task avg 92.57 vs 93.26 (BF16 93.12). Kept IQ3_S as default. With XXS the step is GPU-bound: the +1-4% only
  grows if the IQ2_XXS/IQ2_XS gate/up kernels get the gu-n treatment.

## Combine-wait MALL prefetch 2026-11-23 (DROPPED; tools/sweep32.sh, snapshot/2026-11-23/sweep32.txt, wait-pf-experiment.py)
- STRATA_WAIT_PF=<blocks>: extra blocks in combine_cpu_k10_vec4 read layer l+1's HC/mixer/router/shexp weights while the
  combine waits for the CPU experts, stopping when the flag is up (polled first, then once per chunk). Bitwise (12/12 SAME).
- ms/window (2 rounds): off 18.73 / 18.71; 96 blocks 18.95 / 18.91; 48 18.78 / 18.77; 96 at 8 uint4 18.92 / 18.93.
  Slower every time: the combine rarely waits at 92% VRAM hits (the CPU finishes during the VRAM expert kernels), so the
  blocks only add their launch + one PCIe poll of the mapped flag (~2-4 us a layer). Reverted on the box.
- Conclusion with the 11-22 MALL work: no idle DRAM window exists inside one launch that isn't on the critical path.
  What is left is the ~25 dependent kernels a layer: fewer of them only pays where it shortens that chain.

## GDN step split alone 2026-11-23 (DROPPED; s33_0..3 on the box, 8K, REP 2, alternated)
- gdn_step_norm_multi runs one 512-thread block per head (32 blocks) over a 2 MB state. STRATA_GDN_SPLIT=1 (4 blocks a
  head + gdn_out_norm_kernel) was only tried inside the gfx1151 bundle; alone it is bitwise (SAME) but slower:
  18.50 / 18.50 -> 18.75 / 18.78 ms/window, 114.5 / 114.3 -> 113.0 / 112.8 tok/s. 4x the blocks gives the body
  nothing, the extra launch costs ~7 us a layer. The recurrence is not the GDN gap; a fused last-block-norm split isn't worth building.
- Daily MCLK 1400 now (117.7 / 117.4 plain bench; DECODE_TIMING + REP 2 reads ~114).

## gfx1151 decode switches one at a time 2026-11-23 (DROPPED; snapshot/2026-11-23/sweep34.txt, 8K, REP 2, 2 rounds)
- tok/s, ms/window (base 114.3 / 18.49-18.51): MMVF_ROWS 113.2-113.3 / 18.69-18.71 (bitwise, slower); TSUM 114.2-114.3 /
  18.49-18.51 (bitwise, nothing); ATTN_LANECELL 114.2 / 18.52-18.54 (bitwise, nothing); Q6_PACKED 115.1 / 18.51 but answers
  DIFF: same ms/window, so the +0.7% is a luckier acceptance on changed tokens, not speed. With GDN_SPLIT (above) every
  item of the gfx1151 set is now ruled out alone on gfx1100. SH_STREAM was already out (graph branches cost here).

## Per-layer chain and the fusion budget 2026-11-23 (logs/kprof-k5 on the box, tools/kseq.py; s35_1..4)
- A GDN layer-window is 15 dependent launches: hc read, qkv+z mix, conv_ab, step_norm, out q6k, hc read, router_publish,
  shexp gate/up mix, swiglu_quantize, shexp down IQ4_NL, wait_flag_copy (plan), expert gu, swiglu_q8_1_entries, expert
  down, combine_cpu. QSA layers ~28. ~720-800 launches a window.
- STRATA_EXP_AUTO=2 (IQ3_XXS gate/up through the fused gate/up + SwiGLU + q8_1 kernel, one launch fewer on those layers):
  bitwise, 18.48 / 18.54 -> 18.55 / 18.55 ms/window (114.5 / 114.2 -> 114.0 / 114.1). Level: the launch saved is paid back
  by the fused kernel at 3 tokens. Dropped.
- Budget: an empty node is ~2.8 us (sweep7), so 1 ms needs ~350 launches gone (half the graph) with no slower kernels;
  the persistent GDN megakernel (sweep20) already showed ~0. Fusion rewrites are now expected at +0-3%, not 130 tok/s.

## Tree-draft ceiling 2026-11-23 (patches/mtp-top2.patch, STRATA_MTP_TOP2=1; bench s36d, 8K+32K, REP 2)
- STRATA_MTP_TOP2 never counted: MtpDrafter::record_top2 had no caller and serve takes the all-steps-at-once path.
  mtp-top2.patch calls it after each step and routes TOP2 runs stepwise (diagnostic only; off = untouched path).
- 482 windows: rejected / runner-up right by depth 91/33 (36%), 64/26 (41%), 36/12 (33%) -> a sibling at every depth
  would keep ~0.15 more tokens a window (2.11 -> ~2.26, +7% at best) for up to 3 extra verify rows a window (spec 5/6
  showed wider windows cost about what they keep). Tree drafting: not worth building. Retraining the head is the lever left.

## MTP head experts at higher precision 2026-11-23 (patches/mtp-native.patch, tools/mtp_rt_native.py, `./strata-rdna3.sh mtp4`)
- The shipped MTP head packs its 512 experts at q2_0: relative RMS error 0.42 (q4_0 0.089, q8_0 0.0055). The engine ran
  them through a Q2_0-only kernel (moe_grouped_s2). mtp-native.patch: an rt dir with experts.txt ("<gu type> <down type>")
  holds the experts in the main model's native layout and runs them through native_expert_grouped (q8_1 activations).
  BF16 MTP tensors fetched with tools/mtp_fetch.py (5.2 GB, sha-verified) into data/mtp/bf16.
- VRAM: q2_0 0.71 GB, q4_0 1.42 GB (expert cache 9444 -> 9097 slots), q8_0 2.67 GB (-> 8480 slots).
- Answers change with the head even at FAST=0 (window sizes change the verify numerics), so the 2-prompt bench mixes in
  text luck. Bench (8K / 32K tok/s, 2 rounds): q2 114.3 / 115.4; q8 122.1 / 115.6 (sweep38); q4 119.5 / 111.6 (sweep40).
  FAST=0 (sweep39): q2 101.75 / 107.4, q8 103.1 / 105.95; tokens/window 2.05 / 2.02 -> 2.13 / 2.19 but +1-2 ms/window.
- Varied chats (tools/chatbench.py, 12 requests, 2 rounds, identical both rounds; sweep41): q2 131.1, q4 132.6 (+1.1%),
  q8 130.0-130.1 (-0.8%). Default now q4_0 (MTP=4; MTP=2 = shipped head). Precision is not the head's main limit:
  q8 keeps only +4-8% more tokens. Retraining (distilling to this IQ3_S model) is next.

## Draft-layer options on varied chats 2026-11-23 (tools/sweep42.sh, sweep43.sh; chatbench, 6 prompts a round)
- --mtp-hnorm stream (pre_fc_norm_hidden as one RMS per hyper-connection stream, llama.cpp's qwen4exp MTP graph; the fork
  defaults to one RMS over all 10240): base 131.6 / 130.5 / 131.1 / 131.3 -> 136.1 / 135.9 / 137.1 / 136.7 (+4.3%). The
  pooled norm looks like a transcription slip in the fork; stream is now the default (HNORM=1).
- English draft vocab: +2.6% alone (134.8 / 133.9), 137.2 / 137.0 with hnorm (+0.2%): not taken (non-English loses).
- STRATA_MTP_FULL_HEAD=1 (whole vocabulary): 126.5 / 125.5 (-3.9%): the bigger head costs more than it catches.
- Retrain setup: patches/mtp-dump.patch (STRATA_MTP_DUMP: serve prompt-path residual rows + tokens), tools/mtp_data.py
  (prompts / greedy answers / replay), tools/mtp_pipeline.sh, tools/mtp_train.py (torch reference of the draft layer,
  step-1 + chain-2 distillation, export to the rt format). Torch 2.9.1+rocm6.4 in ~/work/rdna3/train-venv.
- 2026-11-23 MTP retrain result (sweep44): mtp_train.py train --chain 2, 2,000 steps, lr 2e-5, draft-vocab mask
  (labels outside zh60k ignored; with them in, loss was inf). Held-out top-1 per draft step 75.5/69.1/66.2 ->
  81.9/76.2/72.3% (step 1000: 81.1, 1500: 81.8, saturating). Varied chats base 136.1/136.7/136.2 vs ft
  137.1/137.1/137.5 = +0.7%, every round. Offline gain mostly doesn't reach real chats: chatbench prompts differ from
  the generated training prompts, and spec-min-p 0.5 gates on the draft's confidence. Kept (MTP=ft default once built).
  Next: spec-min-p / spec with the ft head (sweep45).
- Box ssh: a remote command whose text contains "strata"/"chatbench" or a pkill pattern gets killed by
  strata-rdna3.sh stop/serve or pkill -f (exit 255). Launch via setsid nohup bash <script>; bracket patterns in pgrep.
- 2026-11-23 sweep45/46 (retrained head): --spec 5 wins varied chats 136.8 -> 137.3 (3/3 rounds) but loses the bench
  (8K 129.6 -> 121.0, 32K 122.7 -> 119.5): dropped. --spec-min-p 0.4 = base. Bench now 129.6 / 122.7 (8K / 32K) with
  q4_0 + hnorm stream + ft head, vs 117.7 / 117.4 before the MTP work (+10% / +4.5%). Perf mode not re-run.

## Draft acceptance, measured per position 2026-11-23 (patches/spec-cumx.patch, spec-dump.patch; tools/ab.sh, spectrace.py, mtp_check.py)
- Box crashed during sweep47 (21:45, amdgpu soft lockup in a vblank irq, rebooted itself). After the reboot the same chats read
  ~133.0 instead of ~137 (clocks, cap, EPP verified as before): all numbers below are relative to that 133.0 base.
- STRATA_SPEC_TRACE now also works in the server (it was CLI-only, which is why sweep47's trace file never appeared): one line a
  round, "T a sfx ms thr p1 .. pn". tools/ab.sh = arms x rounds on the chat prompts (TRACE=1, TIMING=1, HITS=1, SESS=file).
- Default config (MTP=ft HNORM=1 FAST=1 --spec 4 --spec-min-p 0.5), 876 rounds: 2.634 tokens a round, 19.85 ms a round.
  Zero-accept rounds are 25.1% now, not 37% (that was the shipped q2 head, pooled norm): 11.0% no draft offered (p1 < 0.5),
  14.2% offered and rejected. Per draft position (offered / accepted of those reached): 1: 780 / 84.1%, 2: 643 / 83.0%,
  3: 522 / 81.8%. The head's probability is over-confident below 0.9: p 0.5-0.6 holds 37%, 0.6-0.7 48%, 0.7-0.8 59%,
  0.8-0.9 79%, 0.9-1.0 95%. 352 of 876 chains were cut at a draft below 0.5.
- With every draft offered (CUMX run) the live step-1 acceptance is 80.6% and zero-accept 19.5%, 2.82 tokens a round: the offline
  81.9 / 76.2 / 72.3% (18% zero-accept, 2.89 tokens) is what the server gets. No hidden drafting bug.
- --spec-min-p is PER DRAFT (generate.cpp serial loop: T grows while dprob[T-1] >= min_p; the drafter stops its chain at the
  first draft below it, mtp.cpp). The fork has an opt-in fixed cumulative floor (STRATA_SPEC_CUM=S, beside min-p) and an online
  cost policy (STRATA_SPEC_COST=1, sweep47: 136.6 / 137.2 base vs 137.1, level).
- Round ms by window rows (median, no profiling; verify + commit + next draft): 1: 14.95, 2: 16.04, 3: 18.04, 4: 20.93 ->
  rows 2 / 3 / 4 add 1.1 / 2.0 / 2.9 ms (with DECODE_TIMING + VERIFY_PROFILE: 18.33 / 19.44 / 21.16 / 24.05). Each row routes up
  to 10 more experts a layer, so the cost a row grows.
- STRATA_SPEC_CUMX=1 (new, own switch): draft row k stays while prod(p_1..p_k) >= (tokens per ms, decayed) x (ms row k adds,
  STRATA_SPEC_CUMX_ROW_MS=1.1,2.0,2.9); thresholds come out at 0.15 / 0.27 / 0.39. =2 uses the measured acceptance of that
  probability instead of the raw one. ab48c (3 rounds): base 133.0 / 133.0 / 132.5, CUMX=1 134.9 / 133.2 / 134.6 (+1.1%, 3 of 3),
  CUMX=2 133.6 / 133.4 / 136.5 (+1.2%, noisier: it learns its calibration online). Tokens a round 2.63 -> 2.82, ms 19.85 -> 21.1.
  Default now CUMX=1.
- Train/serve mismatches (tools/mtp_check.py: the torch head on the rows the server really drafts from, STRATA_MTP_DUMP_DEC):
  1. FAST=1: the answers were generated with it, but the rows the head trained on came from the prompt path, which has no tail
     skip. Measured on 12 chats (4,616 tokens): decode rows vs prompt-path rows of the same text differ 7.5% (cosine 0.996), and
     the head scores the same on both (77.3 / 70.0 / 64.7 vs 77.5 / 70.3 / 65.2). Not a loss.
  2. Quantization: the server reads the Q8_0 projections as Q4_0 (--mtp-q4 all), q4_0 experts and a Q4_0 head; torch trained in
     float. Float torch picks the server's first draft 91.3% of the time, with all three quantized 94.6% (|dp| 0.053 -> 0.032).
     Cost of the quantization: 77.3 / 70.0 / 64.7 -> 76.7 / 69.0 / 63.3. --mtp-hnorm stream matches by construction.
     mtp_train.py --qat trains through it (straight-through Q4_0).
  3. Vocabulary: training masked to zh60k and the server drafts from zh60k (all 1,943 first drafts inside it): consistent.
     But zh60k cannot emit 4.97% of the tokens the model wrote (ordinary words with ids 25k-100k: "replication", "congestion",
     "PostgreSQL"); every shipped subset (en 40k, rt 106k) misses the same ones.
- The real problem found by the check: on prompts outside its training set the retrained head is NO better than the shipped one
  (77.3 / 70.0 / 64.7 vs 77.3 / 69.8 / 65.7). Round 1's 1,553 prompts were templates (6 code frames x 10 languages x 30 tasks ...),
  its eval set near-duplicates of them: 75.5 -> 81.9% was memorised answers. Round 2 = tools/mtp_round2.sh (self-instruct
  prompts, coding-session continuations, sampled + greedy, thinking on / off, QAT, held-out evals on other prompts / sessions).
- Draft vocabulary (ab49, 3 rounds, tools/draft_vocab_plus.py): every shipped subset holds ids 0-24,999 and a thin pick above;
  the model's ordinary English / code tokens with ids 25k-100k are missing (coverage of the tokens it wrote: check chats 95.05%,
  round-1 training text 96.97%, coding sessions 97.40%). zh60k + the tokens seen in its own round-1 answers + the coding
  sessions' assistant turns = 65,332 ids (tools/draft_vocab_plus.bin): held-out coverage 98.66% (chats) / 99.45% (sessions).
  Varied chats: base 133.1 / 132.4 / 132.4, plus (65k) 135.4 / 136.0 / 138.3 (+3.0%, tokens a round 2.63-2.67 -> 2.70-2.77),
  zh60k + every id < 100,000 (117k ids, 99.9% coverage) 133.9 / 134.6 / 133.8 (+1.1%: the head read doubles). Default now
  DVOCAB=1 (the 65k file; no retraining needed: the draft head is rows of the main head). The next retrain should mask to it.
- Answer text is not a quality check for draft-side switches: two runs of the SAME config can differ after the first few
  tokens (base_1 vs base_2: 0 of 6 answers identical; base_2 = base_3) because the windows differ (FAST's tail skip depends on
  the window), and CUMX makes the windows depend on timing. Draft switches only choose which rows are verified; the teacher-
  forced check (tools/qcheck.sh = quant_quality.py) does not run the drafter at all. Kernel switches: qcheck.sh.
- ab50 (2 rounds, DECODE_TIMING on): old defaults 133.0 / 132.5, new (CUMX=1 + DVOCAB=1) 138.7 / 138.7 = +4.5%: the two stack.
  VRAM expert hit rate on these chats 90.5% (27.5-29.9 hits, 2.9-3.3 CPU entries a layer-window).
  STRATA_FUSE_HITS=1 (gfx12 default: moe_hit_add folded into the combine): 136.6 / 136.5 vs 137.8 / 136.7, 20.85 vs 20.6 ms a
  window: slower on gfx1100, dropped. With CUMX on, two runs of one config can pick different windows (the rule reads the
  measured tokens per ms), so kernel folds are A/B'd with STRATA_SPEC_CUMX=0 (identical windows -> compare ms a window).
- q8_0 on the most-used MTP experts only (tools/mtp_mix.py, mix_ab.sh; snapshot/2026-11-23/mix-ab.txt): routing over 166K cells:
  the top 16 / 32 / 64 / 128 / 256 of the 512 experts hold 18 / 29 / 46 / 67 / 89% of the routing weight (27 never routed).
  Emulated in the q8_0 layout (cold experts = the q4_0 pack's values), tokens kept a round with the new defaults: all q4_0 2.834,
  all q8_0 2.829, 64 hot at q8_0 2.781, 128 hot 2.812 (step-1 acceptance 81.8 / 81.4 / 78.3 / 80.5%: run-to-run spread, no
  trend). A real mixed pack would cost 150 / 300 MiB of VRAM (~70 / ~145 of ~9,100 expert slots) and keep nothing more.
  DROPPED: precision of the head's experts is not what limits it (mtp_check: all Q4_0 costs ~1 point of top-1 in total).
- Expert profile learned from coding sessions (tools/profile_ab.sh; --expert-profile-save over 79 logged session replays, then
  --expert-profile; snapshot/2026-11-23/profile-ab.txt, 3 alternated rounds): held-out sessions (26 requests) VRAM hit rate
  shipped 90.71 / 90.44 / 90.99% vs learned 89.92 / 90.95 / 91.47% (90.7 vs 90.8), 144.3 / 144.5 / 141.5 vs 140.8 / 143.7 /
  149.8 tok/s (the spread is tokens kept a window, 2.99-3.27, not the cache). Varied chats: 90.3 vs 89.8% hits, 138.7 / 135.1 /
  135.2 vs 132.6 / 132.8 / 132.7 tok/s (-2.7%, 3 of 3). DROPPED: the adaptive cache already reaches the same hit rate on
  sessions from the shipped profile, and a session-only profile is worse off its own domain.
- QSA folds (12 layers, patches qsa-kv-pair / qsa-gate-q8 / qsa-idx-pair, all opt-in =1; fork's STRATA_NORM_ROPE=1): chat A/Bs
  cannot resolve them - one config lands on one of a few answer trajectories (ab51: 838 / 858 / 854 windows, 138.7 / 135.6 /
  137.9 tok/s for the SAME kernels; FAST=0 does not remove it, ab52), because CPU and GPU expert arithmetic differ and the
  adaptive cache decides which runs where. On shared trajectories: STRATA_QSA_KV_PAIR (key + value projections in one
  rdna_mmvq_pair launch) -0.02..-0.07 ms a window, answers identical; STRATA_QSA_GATE_Q8 (attention gate + q8_1 quantize, two
  elementwise kernels of one thread shape, in one launch) -0.03..-0.09 ms, answers identical. foldbench.sh (fixed 8K prompt)
  and qcheck.sh are the tools for these.
