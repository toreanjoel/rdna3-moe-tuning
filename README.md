# RX 7900 XTX box — how to run it fast

Current: **`strata-rdna3.sh`** (the RDNA3 Strata fork) — install, serve, tune, bench, all in one script; numbers,
defaults and what was tried are in the first section. Everything after "History" is the earlier stock-Strata
setup and the 2026-11-21 investigation, kept for reference (`snapshot/2026-11-21/`).

## How to run this card: `strata-rdna3.sh` (current)

One script, one way: pull this repo, copy `strata-rdna3.sh` to the box, `install` once, `serve`.
It runs [nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3), an MIT fork of Strata with gfx1100
HIP kernels (WMMA prompt attention, RDNA decode kernels, MTP drafting), pinned at `6c2cd28`.

```bash
scp strata-rdna3.sh bonfire@10.0.0.51:work/rdna3/     # or git clone this repo on the box
ssh -t bonfire@10.0.0.51
~/work/rdna3/strata-rdna3.sh install   # once: apt deps (sudo), clone + build for gfx1100, MTP head, 47 GB expert pack
~/work/rdna3/strata-rdna3.sh serve     # tune card + serve http://127.0.0.1:8080/v1 (OpenAI) and /v1/messages (Anthropic)
~/work/rdna3/strata-rdna3.sh status | stop | tune | bench | uninstall
```

`serve` runs in the background (nohup) and survives logout; no tmux needed. Test:
`curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'`

### Numbers (this box: RX 7900 XTX, Ryzen 9 7945HX, 92 GB DDR5-5200, ROCm 7.2.4, Ubuntu 24.04)

`bench.py`, distinct prompts, 256 generated tokens, median of 5 (2026-11-21):

| server | prompt 8K | prompt 32K | generation |
|---|---|---|---|
| **`strata-rdna3.sh serve` (defaults)** | **2,840** | **3,553** | **102–108 tok/s** |
| same, shipped GEMM table (rejected by ROCm 7.2.4) | 2,641 | 3,266 | 103–104 tok/s |
| fork author's reference (9950X3D2, DDR5-6000) | 2,436 | 3,541 | 114–119 tok/s |
| stock Strata 0.1.62 (history, below) | ~1,900 | ~2,000 | 71–85 tok/s |
| llama.cpp ROCm, `-ncmoe 32 -ub 2048 -fa 1` | 1,112 (2K) | — | 25 tok/s |
| llama.cpp Vulkan, `-ncmoe 22` | 103 (512) | — | 13 tok/s |

### Defaults and limits

| setting | value | why |
|---|---|---|
| power cap | 294 W | hard ceiling, never higher |
| core clock max | 2300 MHz | maxed core = coil whine |
| VRAM clock | 1300 MHz | generation waits on it |
| undervolt | -50 mV | same speed, less heat |
| CPU | EPP `performance`, boost on | the CPU computes the experts not cached in VRAM |
| GEMM table | `gfx1100-hipblaslt-100202.txt` | matches ROCm 7.2.x; +7.5% (8K) / +8.8% (32K) prompts |
| engine config | fork's `rdna3/configs/iq3-1gpu.json` | author's production 1-GPU setup |

Clock/power settings reset on reboot; `serve` and `bench` reapply them (sudo asks only if something differs).

Override per run: `HOST PORT SETUP GPUS TUNE SCLK MCLK VO WATTS EPP BOOST TABLE`, engine overrides
`XENV="K=V ..."` and `XARGS="--flag value ..."`, bench `NAME REPEATS`. Examples:

```bash
HOST=0.0.0.0 ~/work/rdna3/strata-rdna3.sh serve                 # reachable from the LAN
WATTS=250 ~/work/rdna3/strata-rdna3.sh serve                    # quieter
NAME=try XARGS="--spec 3" REPEATS=5 ~/work/rdna3/strata-rdna3.sh bench   # A/B an engine flag
```

### Tried and rejected (2026-11-21, same bench)

| change | prompt 8K / 32K | generation | verdict |
|---|---|---|---|
| `--adapt-swaps 48` / `64` (+GEMM table) | 2,840 / 3,553 | 104.1/101.9, 103.7/107.9 | noise, keep 32 |
| `STRATA_AUX_CPUS=auto` | 2,648 / 3,271 | 101.9 / 98.3 | no gain |
| `--pcie-frac 0.15` | 2,652 / 3,269 | 96.8 / 96.0 | slower |
| `--pool-workers 14` | 2,647 / 3,267 | 101.2 / 100.7 | no gain |

Not tried, by choice: higher clocks or power (limits above), RAM 5600 (board won't run it), `STRATA_ROUTE_TAIL_SKIP`
(+6–15% generation but changes outputs), 2 MiB huge pages (reserves 47 GB permanently, est. 0–5%),
`STRATA_MEGAKERNEL` (author measured 0 gain on 1 GPU). Next real step up: a second 7900 XTX (author: 130–170 tok/s).

### Box layout

```
~/work/rdna3/                  everything; `uninstall` removes it
├── strata-rdna3.sh            this repo's script
├── engine/                    fork source at 6c2cd28, engine/strata, .venv, private .config (~1 GB)
├── data/
│   ├── models/IQ3_S/          Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, 2 shards (82 GB; downloaded by install if missing)
│   ├── packs/iq3_s/           experts.bin (byte copy of the GGUF experts, O_DIRECT reads) + dense pack (48 GB)
│   └── mtp/                   MTP draft head (6.5 GB)
├── logs/                      setup.log, serve.log, serve.out, bench-<NAME>.log
└── bench/<NAME>/              bench.py output per run
```

Local patches the script applies to the fork (`strata-rdna3.sh patch`): `bench.py` runs without the vision encoder,
and `run.py` reads `XENV`/`XARGS`. The fork's `setup.py` moves models of any Strata install in `~/.config/strata`,
so the script gives it its own `XDG_CONFIG_HOME`. Requires ROCm 7.x in `/opt/rocm` and the `render`/`video` groups.

# History — stock Strata on this box (removed 2026-11-21)

Everything below is the earlier stock-Strata setup, kept for reference. The box no longer has it
(`~/.local/share/strata`, `~/work/strata`, llama.cpp removed; old launcher archived in
`snapshot/2026-11-21/box-work-strata.tar.gz`).

## `start-strata.sh`

One-command bring-up of the model server on this box: it applies the GPU and CPU
tuning for the **Radeon RX 7900 XTX (gfx1100)**, patches the Strata engine's draft
flags, then (re)starts Strata. The model answering you right now is the one this
script starts.

## Usage

```bash
./start-strata.sh          # apply settings, restart Strata, show status
./start-strata.sh --plan   # only show what strata launch would do; changes nothing
```

It runs `strata launch --stop` first, so it is also the way to *restart* the server
after editing anything.

## Requirements

- `strata` on PATH, `amd-smi`, `python3`, `sudo` (password or sudoers entry).
- `/sys/class/drm/card*/device/pp_od_clk_voltage` present, which needs
  `amdgpu.ppfeaturemask=0xffffffff` on the kernel command line (GRUB). Without it the
  script falls back to `amd-smi` for clocks only — no undervolt.
- `TUI=on` expects to run inside `tmux` (the Strata dashboard).

## Settings

> **The `Now` column below is this folder's older copy of the script, not what the box runs.**
> The live script is `snapshot/2026-11-21/box-config/start-strata.sh` (sha256 in
> `snapshot/2026-11-21/box-hashes.txt`), which differs: `MAX_SCLK=2300`, `CPU_EPP=performance`,
> `CPU_BOOST=1`, and it adds the `EXP_A` / `EXP_B` switch groups (`./start-strata.sh a|b`).
> Restore from that copy — starting from this stale file changes four settings by accident.

All knobs are at the top of the script; comments carry `stock` (card/Strata default)
and `maxed` (fastest measured run) for each.

| Setting | Now | Maxed | What it does |
|---|---|---|---|
| `POWER_CAP` | 294 W | 402 W | card allows 294–402 W; mostly moot while a clock cap is set |
| `MAX_SCLK` | 2000 MHz | (none, ~2550) | max core clock — drives prompt-reading speed |
| `VOLT_OFFSET` | -50 mV | 0 | core undervolt, 0 to -450; crash/reset = back off one step |
| `MAX_MCLK` | 1300 MHz | 1300 MHz | VRAM clock (stock 1250, up to 1500) — generation waits on it |
| `MODEL` | `iq3_s` | `iq3_s` | quant; iq3_s matches the full model on benchmarks |
| `CONTEXT` | 128k | 128k | costs no expert slots; the agent compacts near 85K |
| `VISION` | off | off | its VRAM is better spent on experts |
| `DRAFTING` | `dflash2` | `dflash2` | speculative drafting; `DRAFT_VOCAB=en` |
| `SPEC` | 3 | 3 | draft depth — 3 beat Strata's default (61.2 vs 56.7 tok/s) |
| `MTP` | off | off | its ~0.7–1 GB of VRAM goes to experts |
| `CPU_EPP` | power | performance | the CPU computes part of every token's experts |
| `CPU_BOOST` | 0 | 1 | 0 = quieter fan, slower generation |
| `STRATA_HIPBLASLT_TABLE` | on | on | gfx1100 prompt-speed table (~600 → ~2000 tok/s at stock clocks) |
| `ACCESS` | local | local | only this PC answers; reach it from the Mac over an SSH tunnel |

Measured, Nov 20: **maxed** (`POWER_CAP=402`, `MAX_SCLK=""`, `VOLT_OFFSET=0`) gave
~2000 tok/s prompt reading at ~350 W and 71–76 tok/s generation at 75K context.
The current settings are the low-power version of that, aiming ~1500 tok/s prompt
reading at roughly 200 W. `MAX_SCLK` is untested since the commit fix — start at 1700,
watch clock and watts during a big read, move ±100.

## What the script actually does

1. **Power cap** — clamps `POWER_CAP` to the card's reported min/max, then
   `amd-smi set --power-cap ppt0`. `amd-smi` prints success even when it refuses, so
   the script greps its output for `must be|not_supported|unable|error`.
2. **Clocks & voltage** — writes `pp_od_clk_voltage` directly: `r` (reset to stock)
   first, then `s 1 <MHz>` (core), `m 1 <MHz>` (VRAM), `vo <mV>`, and finally `c`
   to **commit**. Without `c` the file shows the new values and the card ignores them.
   Waking the GPU (`power/control = on`) first, since a sleeping GPU refuses.
3. **CPU** — `energy_performance_preference` on every core, and `cpufreq/boost`.
4. **Draft flags** — `--spec` / `--spec-min-p` are engine flags, not `strata launch`
   options, so they are written into `strata-<model>.json` under both `args` and
   `user_args`. First edit backs the file up as `.orig`; an empty value restores
   whatever `.orig` had. The config is auto-found under `$HOME` (newest first,
   skipping `docs/`, `bench-results/`, `tests/`, `.git/`, `node_modules/`) unless you
   set `STRATA_CONFIG=`.
5. **Restart** — `strata launch --stop`, patch config, `strata launch`, `--status`.

## Verifying / monitoring

```bash
watch -n 0.5 cat /sys/class/drm/card1/device/hwmon/hwmon*/freq1_input   # clock during a read
amd-smi monitor -g 0                     # watts, temp, vram
strata launch --status
```

The script prints the same checks after applying: the socket power limit and the
`OD_SCLK` / `OD_MCLK` / `OD_VDDGFX_OFFSET` readback.

## Tuning order

1. Set the two reference points: maxed (402 W, no clock cap, no undervolt) or now.
2. Lower `POWER_CAP`, or cap `MAX_SCLK`, until prompt tok/s drops below what you want;
   watch real clock during a big read, not just the setting.
3. `VOLT_OFFSET` in -25 steps for more headroom at the same watts. A crash, freeze or
   driver reset means one step back; a reboot clears it.
4. `MAX_MCLK` in +25 steps while generation tok/s rises; back off if it falls.
5. `SPEC` 2–4 and `DRAFTING` last — drafting mostly moves generation, not reading.

## Gotchas

- Clock/voltage changes need the `c` commit; every run resets to stock first, so
  empty settings really mean stock.
- `SPEC_MIN_P` is MTP-only; DFlash2 uses `--draft-conf` (0.5), set by `strata launch`.
- Several `strata-<model>.json` files are listed and the newest used — set
  `STRATA_CONFIG=` to be sure you are editing the live one.
- With `ACCESS=local` the port is not reachable from the LAN; use an SSH tunnel.

---

# Session state and restore — 2026-11-21 (7900 XTX, `bonfire@10.0.0.51`)

Everything from the decode investigation lives in this folder, so this folder alone is
enough to get the box back to the state measured here.

| path | what it is |
|---|---|
| findings, the measurement trap and the R1–R8 ladder | now in this README, sections below |
| `snapshot/2026-11-21/docs/` | engine `--serve --help` dump, plus the Strata upstream docs and 7900 XTX benchmark reports cited in Sources |
| `snapshot/2026-11-21/box-state.txt` | versions, GPU OD/clock/power read-back, CPU policy, ulimits, RAM |
| `snapshot/2026-11-21/box-config/` | the box's `start-strata.sh` **and** its Strata configs, as they were (plus `launch-iq3_s-128k.live-2026-11-21.json`, the MTP-era launch config that was cleaned up) |
| `snapshot/2026-11-21/tools/` | helpers, also on the box in `~/work/strata/`: `decode_probe.py`, `clocks2.sh`, `clocks.sh` (old, reference only), `ab.sh`, `abx.sh`, `decode_watts.py`, `prefill_watts.py`, `power_guard.sh` (do not run, see Power ceiling), `proc_tree.sh`, `r1-launch.sh` |
| `snapshot/2026-11-21/tools/step.sh` | the reversible ladder driver, also on the box as `~/work/strata/step.sh` — snapshots to `~/work/strata/rollback/`, patches a copy, never restarts (see "Running the ladder yourself") |
| `snapshot/2026-11-21/tools/rt.sh` | one-line remote runner: `rt.sh '<command>'` runs it over key-only SSH on `bonfire@10.0.0.51` |
| `snapshot/2026-11-21/box-hashes.txt`, `CHECKSUMS.txt` | sha256 of the box files and of the snapshot |
| `snapshot/2026-11-21/measurements.csv` | every number in this README in one machine-readable table (clock profile × tok/s × watts × verdict) — the source for any graph |
| `start-strata.sh` (here) | **older** copy: no `EXP_A`/`EXP_B`, `MAX_SCLK=2000`, `CPU_EPP=power`, `CPU_BOOST=0` |
| `snapshot/2026-11-21/box-config/start-strata.sh` | the copy actually running on the box (`MAX_SCLK=2300`, `CPU_EPP=performance`, `CPU_BOOST=1`, EXP_A/EXP_B) |

## Installations made on the box

**No software.** No package, engine, model or config file of yours was installed, edited or
removed. Engine stays `0.1.62-linux-x86_64-hip-rx7900` (0.1.63 is available on the signed feed but was
not fetched). Files added on the box, all new and all mine, in `~/work/strata/`:
`decode_probe.py`, `clocks2.sh`, `ab.sh`, `abx.sh`, `decode_watts.py`, `prefill_watts.py`,
`power_guard.sh` (leave it unrun — see Power ceiling), `proc_tree.sh`, `r1-launch.sh`, plus an
older `clocks.sh` kept only for reference — nothing of yours was overwritten.
`start-mtp.sh` (the temporary `MTP=on` launcher used for R9) was **deleted on 2026-11-21**; the
live server is `bash ./start-strata.sh` with `--mtp off`, verified from `/proc`. `power_guard.sh`
is **not running now**; the clocks are yours. `rt.sh` was **not** added to the box: it runs on
the Mac from `snapshot/2026-11-21/tools/rt.sh` and just wraps `ssh bonfire@10.0.0.51 '<cmd>'`.

**One file added outside `~/work/strata`, and it is not enough on its own.**
`/etc/security/limits.d/60-strata-memlock.conf` (`bonfire - memlock unlimited`) — added with root,
a new file, nothing of yours edited. It raises memlock for login shells (my SSH session reads
`unlimited`) but **not** for the systemd user manager that actually reparents the engine, so the
running engine is still capped at 12,352,954,368 bytes and the PLE table still logs *not locked*.
The complete fix is the `user@.service.d` drop-in in R1 below. Undo the file with
`sudo rm /etc/security/limits.d/60-strata-memlock.conf`.

**One access change, which is a config edit — undo it like this.** To avoid re-typing the
password, `ssh-copy-id` added your **existing** Mac key to the box:

```
~/.ssh/authorized_keys on the box, added line 1:
  256 SHA256:mZuihdmhefkQBGowHt2xCuMeke5oI5M2YdaYuAJP/F8 torean_decloud (ED25519)
  = /Users/toreanjoel/.ssh/id_ed25519.pub (dated Aug 28, your own key, not created for this)
already there before, still needed:
  3072 SHA256:GOj6al8zssy2z5gCoWRGiEE0p4WDKbiSLcO9MnV/Uq4 toreanjoel@Toreans-MacBook-Pro.local (RSA)
```

Undo (removes only the ED25519 line, keeps the RSA one; the local key stays because it predates
this session — delete `~/.ssh/id_ed25519*` yourself only if you want it gone):

```bash
ssh bonfire@10.0.0.51 'cp -a ~/.ssh/authorized_keys ~/authorized_keys.2026-11-21.bak
  grep -v "AAAAIHAOM7mtYSPHSS9uYYQjQA" ~/.ssh/authorized_keys > ~/.ssh/ak.new
  mv ~/.ssh/ak.new ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  echo "left:"; ssh-keygen -lf ~/.ssh/authorized_keys'
```
After that `snapshot/2026-11-21/tools/rt.sh` stops working and password SSH works again.

## Setting changes made, and how to undo them

Only GPU runtime knobs were touched (`pp_od_clk_voltage`, `power1_cap`) — plus the one
`limits.d` memlock file above. They are **already back at your values**, verified by driver
read-back and by `/proc` on the live server:

```
OD_SCLK level 1: 2300 MHz      OD_MCLK level 1: 1300 MHz
OD_VDDGFX_OFFSET: -50 mV       power1_cap: 294 W (card floor is 294, max 402)
CPU: epp=performance  boost=1  governor=powersave   (unchanged, set by your script)
live server: bash ./start-strata.sh -> strata launch --mtp off -> engine 0.1.62-linux-x86_64-hip-rx7900
engine Max locked memory: 12352954368 bytes (11.5 GiB)  -> R1 NOT applied to the server; PLE table
  still cannot lock, and the ~6% in R1 is still on the table
ulimit -Sl / -Hl in a fresh login shell: unlimited (the limits.d file works there, see above)
```

To restore exactly that after any later experiment:

```bash
ssh bonfire@10.0.0.51
cd ~/work/strata && PW=<your sudo password> bash clocks2.sh 2300 1300 -50 294   # prints the read-back
# check nothing of yours drifted on the box (compare against the snapshot hashes):
ssh bonfire@10.0.0.51 'sha256sum ~/work/strata/*'
# if start-strata.sh drifted, put the snapshotted copy back:
scp snapshot/2026-11-21/box-config/start-strata.sh bonfire@10.0.0.51:work/strata/
ssh bonfire@10.0.0.51 'cd ~/work/strata && ./start-strata.sh'                       # ~17 min reload
```

Configs that must never be deleted, all snapshotted here. Exact restore, one line each
(the snapshot renames only the launch config — note the different name on the right):

```bash
B=snapshot/2026-11-21/box-config
scp $B/start-strata.sh      bonfire@10.0.0.51:work/strata/start-strata.sh
scp $B/run-iq3_s.sh         bonfire@10.0.0.51:.local/share/strata/install/run-iq3_s.sh
scp $B/strata-iq3_s.json    bonfire@10.0.0.51:.local/share/strata/install/strata-iq3_s.json
scp $B/strata-iq3_s.json.orig bonfire@10.0.0.51:.local/share/strata/install/strata-iq3_s.json.orig
scp $B/strata-iq3_s.json.bak  bonfire@10.0.0.51:.local/share/strata/install/strata-iq3_s.json.bak
# snapshot name launch-iq3_s-128k.json, box name iq3_s-128k.json:
scp $B/launch-iq3_s-128k.json bonfire@10.0.0.51:.local/state/strata/launch/iq3_s-128k.json
```

**Verified at end of session, no restart needed:** `strata-iq3_s.json` = `7e2177b6…` (matches the
snapshot) and `.orig` = `4238e80b…` (never touched). The live launch config carried **two leftovers
from the MTP test** — `"mtp": true` and an extra `--spec-min-p 0.5` in `args` — which the normal
launcher does not rewrite, so the snapshotted copy was put back (verified `c4644cdd…`, the running
engine untouched at pid 68818). The leftover-bearing file is kept here as
`box-config/launch-iq3_s-128k.live-2026-11-21.json` if you want to compare.

```bash
B=snapshot/2026-11-21/box-config
# to undo that cleanup (put the MTP-era config back):
scp $B/launch-iq3_s-128k.live-2026-11-21.json bonfire@10.0.0.51:.local/state/strata/launch/iq3_s-128k.json
```

Then verify against `snapshot/2026-11-21/CHECKSUMS.txt` (left column) and
`box-hashes.txt` (what the box holds), and reload with `./start-strata.sh`.

## Keeping the engine pinned at 0.1.62

Pinned component: `~/.local/share/strata/components/engine/0.1.62-linux-x86_64-hip-rx7900/`
(signed manifest `engine-component.json`, variant `rx7900`, tuning table
`tuning/gfx1100-hipblaslt-100200.txt`). Nothing updates itself: `components/.updates/state.json`
has shown `engine: 0.1.63, engine_installed: 0.1.62` for two days with no install, because
`strata update` / the app's Update button is the only thing that switches. So:

- run `strata update --check` when you want to know, and plain `strata update` never by habit;
- if you want the feed unreachable while experimenting, point it at a folder that does not
  exist (`STRATA_UPDATE_FEED=$HOME/no-feed`) — update checks then fail instead of fetching
  (only affects processes that see that variable; it is not how the engine is normally started).
- the engine binary itself is not in a config — if a step (R8 in the Ladder section) is ever applied, the
  previous component dir is what you restore, so don't delete `components/engine/0.1.62-linux-x86_64-hip-rx7900`;
  `strata doctor` and `strata components` read-only-check it.

Two boot-level facts the restore needs (both already on the box, recorded in `box-state.txt`):
`amdgpu.ppfeaturemask=0xffffffff` is in `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub` —
that is what makes `pp_od_clk_voltage` writable, and after any GRUB change `update-grub` + reboot;
and sudo is plain password sudo (`(ALL : ALL) ALL`), **no** NOPASSWD entry was added.

## What is actually running

- Model IQ3_S Qwen3.8-Flash-Next, 128K ctx, `--kv int8 --kv-resident 32768`,
  `--expert-cache auto`, `--prefill auto`, `--spec 3 --spec-min-p 0.5`, dflash2 drafting, MTP off.
- Live env: only `STRATA_HIPBLASLT_TABLE=on`. Your script on the box already sets
  `CPU_EPP=performance`, `CPU_BOOST=1`, `MAX_SCLK=2300`, `MAX_MCLK=1300`, `VOLT_OFFSET=-50` — the
  `power`/`boost=0` pair in this folder's older copy is not what runs. `STRATA_NO_ACQUIRE=1` comes from the launcher.
- Startup log: 8438 VRAM expert slots pre-filled from `expert-profile.bin`; expert arena
  46.84 GiB in RAM; **882 MiB of VRAM free with everything loaded**; PCIe probe 28.4 GB/s
  → **`pcie_frac 0.55` (auto, never retuned for gfx11)**; **15 CPU pool workers + host thread**
  (default, = one per physical core).
- Per-request stats in `~/.local/state/strata/launch/iq3_s-128k.engine.log`:
  decode expert-cache hit **88–92%**, 5.3–7.9% of routed experts fetched over PCIe,
  the rest computed on the CPU; KV streaming 99.85% VRAM hits; draft acceptance 78–83%.

## Measurement (and the trap)

Probe: fixed ~28K-token prompt (prompt cache hit) + 300 generated tokens, then read the
engine's own `/v1/status` `last_timings`. Script: `~/work/strata/decode_probe.py`.

| profile | decode tok/s (median) |
|---|---|
| first cold run at 28K | 60–72 |
| warm, your current clocks (s2300 m1300 vo-50 294 W) | 83.5 |
| s2500 m1400 402 W | 84.6 |
| s2300 m1400 294 W | 84.8 |
| s2100 m1400 294 W | 83.9 |
| s2500 m1500 294 W | 84.3 |

No watts column: 294 W is the lowest cap the driver accepts, so all rows but the 402 W one are
already at that minimum, and mean draw measured 278–311 W in every case. See "Power verdict"
below the ladder.

Two corrections to my own first readings:

1. **The apparent +15% from clocks was warm-up, not clocks.** The expert-cache tier and the
   prompt cache settle over the first minutes; once warm, all clock profiles land 83–85 tok/s.
   Any real A/B must **alternate** configs (helper `~/work/strata/abx.sh`) — never measure A then B.
2. **Server concurrency is 1.** The box also serves the model that answers you here, so a chat
   in flight while a probe runs drags the probe down. Idle the chat before measuring
   (`decode_probe.py` now waits for `in_flight == 0`).

Also learned: the OD clock write must go through **one** sudo shell — separate `sudo tee` calls
silently lose writes (`~/work/strata/clocks2.sh` does it right and prints the driver read-back).
The power floor is 294 W (`power1_cap_min`), so "lower than 294 W" is only reachable by
capping sclk/mclk, not by a watt limit. OD mclk max is 1500 MHz.

## Free wins sitting in the log (no power cost, one restart each)

1. **PLE table is not locked in RAM.** Log: *"the PLE table (26.8 GiB) was not locked: this
   program may lock only 11.5 GiB (RLIMIT_MEMLOCK)… the lock was measured up to 6% faster in
   decode (#4671)"*. Measured on the box: `ulimit -Sl` **and** `ulimit -Hl` are both
   12063432 KB (11.5 GiB), so `ulimit -l unlimited` **as your user cannot work** — you cannot
   raise past the hard limit. Real fixes, pick one:
   - new drop-in file `/etc/security/limits.d/60-strata-memlock.conf` containing
     `bonfire - memlock unlimited`, then log out and back in (adds a file, changes none of yours).
     Caveat: `limits.d` is applied by PAM at **login**, so it only covers Strata started from your
     session — a Strata started by systemd, `systemd-run`, at boot or over a non-PAM path ignores
     it; for those use `LimitMEMLOCK=infinity` on the unit; or
   - start it once through systemd-run so it keeps running as **bonfire** but with the raised limit:
     `sudo systemd-run --uid=bonfire --gid=bonfire --property=LimitMEMLOCK=infinity \`
     `--working-directory=$HOME/work/strata --collect -p StrataDecode ./start-strata.sh`
     (state, config and logs stay under bonfire's `~/.local`; no root-owned files).
   Do **not** simply run `sudo ./start-strata.sh`: the engine would run as root, use root's home
   for its state/config/logs and leave root-owned files behind. `sudo prlimit
   --memlock=unlimited:unlimited --pid <pid>` can raise a live process's limit, but the PLE table
   is locked during load, so it only helps if applied **before** the engine maps it.
   RAM check first: 26.8 GiB locked + 46.8 GiB arena ≈ 74 of 92 GiB.
   Verify with `grep "PLE table" iq3_s-128k.engine.log` → should say *locked*.
2. **`adaptive overlap: off (STRATA_ADAPT_OVERLAP=1 turns it on)`** — one `--set`.

## Running log — what each step actually did

All numbers are `decode_watts.py` at **1000 generated tokens** (the honest length) with the
28K-token prompt, alternating with `decode_probe.py` at 300 tokens. Baseline before any
restart: **68.0 tok/s** at 1000 tokens (177 W mean / 184 W max), 69.4 at 300 tokens.

| step | what happened | decode 1000 tok | decode 300 tok |
|---|---|---|---|
| baseline | as configured on 2026-11-21 | 68.0 | 69.4 |
| R1 relaunch (`strata launch` under `systemd-run -p LimitMEMLOCK=infinity`) | the engine still starts at 11.5 GiB memlock (see below) | 64.2 | 75.5 |
| R3 `./start-strata.sh a` (EXP_A, 13 decode switches confirmed present in `/proc/<engine>/environ`) | **neutral** | 68.5 | 76.4 |
| R1 memlock unlimited, lock handed back | 67.3 — the 6% did not appear; RAM pressure released the lock | 67.3 | — |
| 0.1.63 + experts (R8 after R6) | **+4.4% on the session baseline, inside 280 W** | 71.0 | 74.9 |
| R9 `MTP=on` (temp `start-mtp.sh`, since deleted) | **no gain**: 67.2 at 1000 tok / 75.5 at 300 tok, 174 W mean / 183 W max | 67.2 | 75.5 |
| your profile `s2300 m1300 vo-50` | decode fine; prompt peak 262 W, inside the 280 W ceiling | 65–70 | 83–85 |
| static `s1500 m1300 vo-50` | whine-free fallback, prompt peak 182 W | ~46 | — |

Two things that fell out of it:

- **R1 does not work by itself.** `limits.d` raises memlock for *login shells*, but
  `strata launch` hands the runtime to the **user manager** (`systemd --user`, PID of
  `/usr/lib/systemd/systemd --user`), and that process holds `Max locked memory
  12352954368` (11.5 GiB); anything it spawns inherits it, so the engine logs the same
  "not locked" line no matter what your shell says. Real fix (needs root and a session
  restart, not done yet): a drop-in for the user manager —
  `/etc/systemd/system/user@.service.d/60-strata-memlock.conf` with `[Service]` /
  `LimitMEMLOCK=infinity`, then `sudo systemctl daemon-reload` and restart the user session
  (or reboot). Note the log line never re-appears as "locked" until that is done, so it is
  also the check that the fix worked.
- **R9 (MTP drafting) is neutral here, contrary to the docs' 1.6–1.8× claim.** At 1000 tokens
  67.2 tok/s with `MTP=on` vs 68.0 baseline, same watts (174 mean / 183 max), and the launch
  config reverted to `--mtp off` afterwards. Drafting is not the missing 30 tok/s either.
- **Decode is not context-limited here.** 2K → 8K → 28K context measured 73.5 / 72.2 /
  76.3 tok/s. So expert-cache pressure and KV streaming are not the wall at these lengths;
  the fixed per-token pipeline (GPU kernels + drafting) is. That makes R9 (drafting) and
  kernel-side switches the only remaining big levers, and predicts that R6 (expert-cache
  shape) and R7 (`pcie_frac`) will do little.

`rocprofv3` is installed (`/opt/rocm/bin/rocprofv3`, plus `rocprofv3-attach`), but attaching
needs the engine to have been started with the attach library loaded, so the GPU-busy-%
profile still needs one restart with that env set.

## Ladder, cheapest first

Each engine change needs a restart: **load takes ~17 min** (`engine_ready_s: 1008`) and it
drops the tunnel this conversation uses. One restart per step, otherwise the A/B means nothing.
**R2 is worse: `strata engine calibrate --measure` wants an exclusive GPU, so the chat model
this conversation runs on is down for the 15–30 min sweep *plus* the ~17 min reload.** Do it
when you do not need the model.

| step | change | why | cost |
|---|---|---|---|
| R1 | raise memlock where the engine is actually born: `/etc/systemd/system/user@.service.d/60-strata-memlock.conf` = `[Service]` + `LimitMEMLOCK=infinity`, then `sudo systemctl daemon-reload` and restart the user session (reboot is simplest). The `limits.d` file already on the box is **not** enough — verified: the live engine still shows 11.5 GiB | log says up to 6% decode, no watts; free of charge inside the 280 W budget | root + restart |
| R2 | `strata engine calibrate --profile launch-iq3_s-128k --measure`, then `--review-result` and `--apply-result` (each has a matching `--rollback-result`) | its own decode sweep; sets pcie/pool values instead of us guessing them | exclusive GPU, 15–30 min + reload |
| R3 | `./start-strata.sh a` (your own EXP_A: `STRATA_EXPERT_V2`, `STRATA_TSUM`, `ATTN_LANECELL`, `Q8_PACKED`, `MMVF_ROWS`, `LFUSE`, `QFUSE`, `GDN_*`, `PLE_BATCH`) | decode kernels Strata enables for Strix Halo/RDNA4 but not the 7900 XTX | restart |
| R3b | after EXP_A, before keeping it: score it, not just speed — `~/work/strata/decode_probe.py` prints `draft N/N (%)`; compare that and read a few real answers (code + long context) against the R2 baseline | EXP_A changes kernels, so acceptance and answer quality must match; a kernel that is fast and wrong is a loss | probe only |
| R4 | `--set STRATA_ADAPT_OVERLAP=1` | log-confirmed off | restart |
| R5 | SPEC A/B: `./start-strata.sh` with `SPEC=2` vs `SPEC=3` (your script writes `--spec` into the config and keeps `.orig`) | each verify step loads experts for every drafted token; when the CPU/PCIe path is the limit, 2 can beat 3 despite lower acceptance | restart (or config edit + reload) |
| R6 | expert-cache shape: `--expert-cache N` above 8438, and `--expert-cache-per-layer` (docs: 2.97% → 21.4% hits at small slot counts) | hit rate is 88–92%; only 882 MiB VRAM free, so pair it with a smaller `--kv-resident` | restart |
| R7 | `--pcie-frac` (`./step.sh apply pcie` writes 0.35; `STRATA_PCIE_FRAC_DEFAULT` only takes on/off/old/auto; auto here is 0.55, gfx12 ships 0.15) — **only if R2's calibrate leaves `pcie_frac` alone**; check the log line `PCIe probe: … -> pcie_frac` after R2 first | 5–8% of experts arrive over PCIe | restart |
| R8 | `strata update` to engine 0.1.63 | 0.2.17 feed: route-tail-skip now off by default, RAM floor 24 → 17 GiB (more experts resident) | own restart, last |
| R9 | ~~MTP draft layer~~ **done, measured neutral** (67.2 vs 68.0 at 1000 tokens, same watts); the temp launcher `start-mtp.sh` was deleted and `--mtp off` is back | ruled out: it is not the route to 100+ tok/s | — |

## Path to 100+ tok/s (and it cannot come from clocks)

You asked for 100+ tok/s on average, at ≤280 W, and never with SCLK maxxed — that is what makes the
coil whine. Those three pin the problem: at `MAX_SCLK=2300` generation draws 176–183 W and reaches
only 65–70 tok/s on long answers, while every clock above that buys ~1 tok/s at 350–402 W and inside
the whine band. **So the speed has to come from memlock, expert-cache residency and kernels.**
Watts stopped being the binding constraint when the ceiling moved to 280 W: `s2300 m1300 vo-50`
peaks 262 W on prompt reading, which fits (see Power ceiling below).

Where we are, measured at 1000 generated tokens (the number that reflects real answers):
**65–70 tok/s** at 28K context, expert-cache hit 88–92%, draft acceptance 75–80%.

Expected contribution of each step, with how to confirm it. R1 is the only one that is free; R3 and
R9 are now measured, not hypotheses:

| step (in `./step.sh` order) | expected | how it is confirmed |
|---|---|---|
| R1 memlock — **done, and it did NOT pay**: memlock is now unlimited and the engine *locks* the 26.8 GiB table in 1.1 s, then hands the lock back (`PLE table lock given back: the lock left 13.5 GiB of RAM for the conversation cache, 40.3 GiB without it`), measured **67.3 tok/s vs 68.0** at 173 W. The box has **92 GB RAM** (`free -g` total 92, 55 used — my earlier "24 GB" note was wrong), so the lock is not RAM-starved in principle: 0.1.62 gave it back at a moment when only 13.5 GiB was free, so re-try with less loaded before giving up | ~~`apply ccache`~~ (`--conversation-cache-mib 4096`) tested: the flag reached the engine but the lock was still handed back, scoring **64.9** — rolled back |
| R2 `engine calibrate` | unknown — it *measures* pcie/pool, so do it before hand-setting anything | `PCIe probe: … -> pcie_frac` log line |
| R6 expert-cache shape — **`apply experts` done: +1.6% (69.1 at 1000 tokens) but the hit rate jumped to 94–96.6% from 88–92%, so cache hits are no longer the wall**; `apply perlayer` still untried and now has little headroom (bare `--expert-cache-per-layer`, `engine_help.txt:224`: 2.97% → 21.4% hits at 8 slots/layer, 70.4% at 64) | the 1000-token vs 300-token gap (68 vs 84) is cache drift, so this is where ~20% lives: hit 88→95% ≈ +8–10% (community 7900 XTX: 53.8%→44 tok/s, 86.1%→70.5) | engine log `expert cache: N hits / M lookups` |
| R7 `pcie_frac` — `apply pcie` writes `--pcie-frac 0.35` (or `apply pcie=0.15`); `STRATA_PCIE_FRAC_DEFAULT` only takes on/off/old/auto (`SETTINGS.md:166`, `AMD_HIP.md:422,530`) and an explicit `--pcie-frac` always wins. Only after R2 if calibrate leaves it alone | fewer experts fetched over PCIe at the probed 28.4 GB/s (auto here is 0.55) | `PCIe probe: … -> pcie_frac` log line |
| R4 ADAPT_OVERLAP | a few %; Strata only enables `SH_STREAM` when it wins >3% | `decode_watts.py` at 1000 tokens |
| R5 SPEC 3 → 2 | a verify step, not a speed step | draft-acceptance % in the engine log |
| R8 engine 0.1.63 — **done: 71.0 tok/s at 1000 tokens (+2.8% on 0.1.62+experts, +4.4% on the session baseline), 176 W**, plus `R4 hit path ON - resident experts are computed on the GPU` and `token graph hit path: 9463 resident experts` | ~~biggest untried lever: ~7 GiB more RAM for resident experts, which feeds R6's hit rate | `strata --version`, then the hit-rate line |
| ~~R3 EXP_A decode kernels~~ | **measured neutral**: 68.5 vs 68.0 at 1000 tokens | done |
| ~~R9 MTP drafting~~ | **measured neutral**: 67.2 vs 68.0 at 1000 tokens, same watts | done, reverted |

Honest arithmetic: 68 × 1.06 (R1) × 1.10 (R6) ≈ **79**, and getting to 100 needs 0.1.63's residency
gain plus whatever calibrate finds. The box already does **83–85 tok/s at 300-token answers**, so
100+ "on average" is plausible for short/medium answers once the cache stops drifting, and is **not
yet shown** for 1000-token answers. Judge everything at 1000 tokens; if the ladder ends near ~85–90,
the remaining gap is engine-side (0.1.63 kernels), not something these switches can buy.

Stacked optimistically that is 1.3–1.6×, i.e. **85–110 tok/s** — so 100+ is within reach of
this silicon but no single switch guarantees it. External evidence that gfx1100 has headroom
(a different model and engine, so only indicative): Qwen3.6-35B-A3B decodes ~85 tok/s on
llama.cpp HIP, ~103 on hipEngine, ~128 on llama.cpp Vulkan — HIP decode on RDNA3 is known to
lag (llama.cpp issues #13565, discussion #15021).

### Running the ladder yourself, with rollback

`~/work/strata/step.sh` (copied from `snapshot/2026-11-21/tools/step.sh`) is the reversible driver.
It **never restarts the server and never edits your `start-strata.sh`**: it snapshots the launcher,
the model config and the launch config into `~/work/strata/rollback/<timestamp>/`, patches a copy
(`start-step.sh`) plus the config, then prints the one line you run when you choose to take the
~17 min reload. One order, used the same way in the Order note and in `./step.sh list`: **memlock (R1) → calibrate
(R2) → experts + perlayer (R6) → pcie (R7) → overlap (R4) → spec2 (R5) → update (R8) last, and 0.1.63 is
never mixed into an A/B of anything else.**

```bash
cd ~/work/strata
./step.sh status                     # current engine: memlock, --set env, PLE line, pcie_frac
#   (the launcher itself refuses to start if memlock is still capped; MEMLOCK_OK=1 overrides)
./step.sh apply memlock              # prints the 4 sudo lines; nothing changes until you run them
./step.sh apply experts              # --expert-cache 12000 --kv-resident 24576 in BOTH strata-iq3_s.json
#                                        and the live launch config iq3_s-128k.json (each saved as .stepbak)
./start-step.sh                      # <- your call: this is what drops the model
./step.sh probe 1000                 # decode_watts + decode_probe at N tokens, 280 W budget check
./step.sh rollback                   # back to the FIRST snapshot: launcher, model config, launch config
./start-strata.sh                    # then you are exactly where you were (verified by box-hashes.txt)
```

Two details worth knowing, both tested on the box: `apply` snapshots first, and the rollback
baseline is written **only by `./step.sh save` or the first `apply`** — later applies keep audit
snapshots in `rollback/` but never move the baseline, and snapshot dirs are timestamp+PID unique so
two applies in the same second cannot overwrite each other. Verified end to end: after
`apply experts` + `apply overlap`, `rollback` returned both configs byte-identical to this snapshot
(`iq3_s-128k.json` = `c4644cdd…`, `strata-iq3_s.json` = `7e2177b6…`).

**No service, no auto-start — your one script still starts it.** The systemd `user@.service.d`
drop-in is only for a server started by systemd, and you don't have one. Your terminal's
`ulimit -l` = `12063432` because it was opened **before** `limits.d/60-strata-memlock.conf` was
added; a *new* login already reads `unlimited` (verified over SSH). So: close that terminal, open
another, run `./start-strata.sh` — nothing else changes. I added a guard at the top of your script
(`add-memlock-guard.sh`, original kept on the box as `start-strata.sh.pre-memlock` and archived here
as `box-config/start-strata.sh`) it now raises the limit **first**, then checks the result: if a new
terminal *still* shows 12063432 KB the launcher prints
`!! memlock still 12063432 KB - open a NEW terminal (limits.d only applies to new logins) and run this again.`
and **refuses to load** (`exit 1`) rather than burning ~17 min on a capped engine — `MEMLOCK_OK=1
./start-strata.sh` loads capped on purpose. Undo: `cp start-strata.sh.pre-memlock start-strata.sh`.

`memlock` is the one step needing root, and it is also the one that survives everything:

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/60-strata-memlock.conf
sudo systemctl daemon-reload && sudo systemctl restart user@$UID.service   # or reboot
# undo: sudo rm /etc/systemd/system/user@.service.d/60-strata-memlock.conf && sudo systemctl daemon-reload
```

**One profile decides where the remaining effort goes** (run it while nothing else is
connected, ~1 min of generation):

```bash
cd ~/work/strata
( rocprofv3 --kernel-trace --stats -o /tmp/prof -- sleep 60 ) &
python3 decode_probe.py 1 1000 112 profiled; wait
# GPU busy % = total kernel time in the stats file / 60 s
```
Under ~60% busy means the GPU waits on CPU/PCIe experts → R1, R2, R6, R7 matter most.
Over ~85% means the kernels themselves are the wall → R8 (the 0.1.63 decode kernels) matters most;
R3 and R9 are already measured neutral, so nothing else in that direction is left to try.

Skip for now: hipBLASLt table work (`tune_hipblaslt`, table `gfx1100-hipblaslt-100200.txt` is
already in the bundle) and `EXP_B` — both are prompt-reading, not decode.

### Power ceiling — 280 W max, and SCLK must not be maxxed (coil whine)

The real constraint, as restated by you on 2026-11-21 — *"i want to avoid the coil wind, that is my
issue which happens if SCLK is maxxed out, that is why and I dont mind 280w max but I want 100+ tps
on average"* — is **≤280 W and never SCLK maxxed**. So the watt ceiling is not the binding problem;
the whine rule is, and it forbids the only setting that ever measured faster than ~85 tok/s.
(Everything earlier in this file that says "200 W is a hard total" was written before that
restatement; `measurements.csv` now carries both verdicts, 280 W and the stricter 200 W one.)

Measured with `snapshot/2026-11-21/tools/decode_watts.py` and `prefill_watts.py`, which sample
the card at 5 Hz and keep only samples while a request is in flight — so decode and prompt
reading are measured separately, and idle/prefill no longer blur the number. That is what the
earlier "290–311 W" rows really were: prompt reading, not generation.

| profile (`MAX_SCLK MAX_MCLK`) | decode tok/s | decode mean / max W | prompt read tok/s | read mean / **peak** W |
|---|---|---|---|---|
| s2300 m1300 (your current) | 65–70 | 176 / 183 W | 1684 | 226 / **262 W** |
| s2300 m1250 (stock mclk) | 68 | 177 / 187 W | — | — |
| s2000 m1300 | 61 | 162 / 170 W | 1492 | 201 / **233 W** |
| s1750 m1300 | 57 | 156 / 176 W | 1342 | 187 / **222 W** |
| s1600 m1300 | 52 | 149 / 154 W | 1224 | 174 / **225 W** |
| s1500 m1300 | 46 | 132 / 133 W | 961 | 149 / **182 W** |
| your script's "maxed" (no cap, 402 W) | ~84 at 300-token runs | ~290–311 W during reads | ~2000 | ~350 W |

**Verdict at 280 W: your current `s2300 m1300 vo-50` profile already fits.** Decode 176–183 W, the
worst prompt read seen 262 W, and nothing else measured went faster. The card refuses any
`power1_cap` below 294 W, so watts can only be held with clock caps, and capping lower only costs
speed — `s1500` is the whine-free emergency setting (prompt peak 182 W, decode ~46 tok/s), nothing more.

What is out on **both** counts — watts and whine — is the uncapped setting (`MAX_SCLK=""`,
`POWER_CAP=402`): 350–402 W during reads, i.e. exactly the maxxed-SCLK state that makes the coil
sing, for ~1 tok/s over s2300. Keep `MAX_SCLK=2300`. If the whine shows up, chase it without losing
decode: `VOLT_OFFSET` `-50 → -75` (one step per run, crash = go back a step), and `EXP_B`
(`./start-strata.sh b`) moves prompt reading onto the matrix cores, which is the load that whines.
`tools/power_guard.sh` is unnecessary at 280 W and stays withdrawn — it reacts seconds late, which
is tolerable for a soft average and useless against a ceiling. If you ever run it, stop it with
`pkill -f 'power_gua[d].sh'`, never `pkill -f power_guard.sh` over SSH (it matches the SSH command
and drops your session).

**So the missing 30 tok/s cannot come from clocks** — it has to come from memlock, expert-cache
residency and decode kernels. That is the step kit below.

**Status as this session ends: the box already sits inside 280 W and never maxxes SCLK** —
`MAX_SCLK=2300`, prompt-read peak 262 W, decode 176–183 W, `POWER_CAP=294` (the card's floor).
Nothing to undo here; `s1500 m1300` (prompt peak 182 W) is only the whine-free fallback, and the
uncapped 350–402 W state is the one to stay away from on both watts and noise.

Longer answers are slower than short ones at the same clocks: 300 generated tokens measured
83–85 tok/s, 1000 generated tokens measured 65–70, because the VRAM expert cache drifts as the
answer goes on (hit rate 88–92%). Judge any change at ~1000 tokens, not 300.

### Power verdict, said plainly

Once the server is warm, clocks only move decode between about **83.5 and 85 tok/s**
(your 294 W profile 83.5; s2500 m1400 at 402 W 84.6) — measured at 300-token answers, and
that spread vanishes at 1000 tokens. So **there is no decode speed worth buying with watts** — and
the whine rule makes the trade worse still, because the clocks that would make the coil sing buy
about **1 tok/s**. `MAX_SCLK=2300` stays; the hunt for 100+ is the kernel/cache ladder, not the
clock table.

### Order note

One order, identical to "Running the ladder yourself" and to `./step.sh list`:
**R1 memlock → R2 `engine calibrate` → R6 experts + perlayer → R7 pcie → R4 overlap → R5 spec2
→ R8 update (0.1.63) last, never mixed into an A/B** (identical to `./step.sh list`). EXP_A (R3) and MTP (R9) were measured **neutral** and are off the
list. Calibrate is R2 rather than later because it *measures* pcie/pool settings, so hand-setting
them afterwards would be guesswork — and R7 is therefore only a fallback if calibrate leaves them
alone.

## The numbers, drawn (all from `snapshot/2026-11-21/measurements.csv`)

```mermaid
xychart-beta
  title "decode tok/s vs core clock — m1300 vo-50, 1000-token answers only"
  x ["1500", "1600", "1750", "2000", "2300"]
  y "tok/s" 40 --> 75
  line [46, 52, 57, 61, 68]
```

```mermaid
xychart-beta
  title "prompt-read peak watts vs core clock (280 W ceiling; s2300 peaks 262 W, uncapped 350 W does not fit)"
  x ["1500", "1600", "1750", "2000", "2300", "uncapped"]
  y "watts" 150 --> 400
  bar [182, 225, 222, 233, 262, 350]
```

```mermaid
xychart-beta
  title "decode mean watts vs core clock (m1300) — 45 W buys 22 tok/s; the uncapped row has no decode
       figure because the 290-311 W seen there was prompt reading, not generation"
  x ["1500", "1600", "1750", "2000", "2300"]
  y "watts" 120 --> 200
  line [132, 149, 156, 162, 177]
```

```mermaid
xychart-beta
  title "300-token answers only: clocks above the ceiling buy 1.1 tok/s (s2500/m1400 = 402 W)"
  x ["s2300 m1300 (262 W peak)", "s2500 m1400 (402 W)"]
  y "tok/s" 80 --> 90
  bar [83.5, 84.6]
```

```mermaid
xychart-beta
  title "session A/B ladder at 1000 tokens — baseline 68.0, nothing gained (R1 failed, R3 and R9 neutral)"
  x ["baseline", "R1 memlock", "R3 EXP_A", "R9 MTP"]
  y "tok/s" 60 --> 75
  line [68.0, 64.2, 68.5, 67.2]
```

```mermaid
xychart-beta
  title "answer length is the hidden variable: same clocks, 300 vs 1000 tokens"
  x ["300 tokens", "1000 tokens"]
  y "tok/s" 60 --> 90
  bar [83.5, 68.0]
```

## Files that must not be touched

- `~/.local/share/strata/install/strata-iq3_s.json` + `.orig` + `.bak`
- `~/.local/state/strata/launch/iq3_s-128k.json` (what the running server actually uses)
- `~/.local/state/strata/launch/iq3_s-{32,64,96}k.json`, all `*.engine.log`
- `~/work/strata/start-strata.sh` — your script already keeps `.orig` backups when it edits
  `--spec`; test from a copy, e.g. `cp start-strata.sh start-strata-test.sh`.

## Helper files I added (nothing else)

- `~/work/strata/decode_probe.py` — fixed-context decode measurement, waits for an idle server
- `~/work/strata/clocks2.sh` — apply clocks/voltage/power in one sudo call + driver read-back
- `~/work/strata/ab.sh`, `abx.sh` — one cell / interleaved A/B with mean-watts sampling
- `~/work/strata/decode_watts.py`, `prefill_watts.py` — decode vs prompt-read watts, in-flight only
- `~/work/strata/proc_tree.sh` — who actually spawned the engine (this is how R1 was diagnosed)
- `~/work/strata/r1-launch.sh` — the memlock relaunch attempt; **kept as the negative result**, the
  engine still inherits 11.5 GiB through the user manager
- `~/work/strata/start-mtp.sh` — temporary `MTP=on` launcher, **deleted at the end of the session**
- copies of all of them in `snapshot/2026-11-21/tools/`, every measurement in
  `snapshot/2026-11-21/measurements.csv`, the engine `--serve --help` dump in
  `snapshot/2026-11-21/docs/engine_help.txt`, and the Strata upstream docs I read in
  `snapshot/2026-11-21/docs/strata-upstream/` and `…/strata-bench/`

Sources: your `start-strata.sh` and the live engine log; Strata docs
`docs/AMD_HIP.md`, `docs/AMD_HIP_PERFORMANCE.md`, `docs/HOW_IT_WORKS.md`, `docs/SETTINGS.md`,
`docs/bench-results/2026-10-06-community-rx-7900-xtx/README.md` (7900 XTX decode tracks the
expert-cache VRAM hit rate: 53.8% hit → 44 tok/s, 86.1% hit → 70.5 tok/s) and the v0.2.16 /
v0.2.17 release notes.

## Measuring again (no risk, no restart)

```bash
cd ~/work/strata
python3 decode_probe.py 4 300 112 label      # median decode tok/s, waits for an idle server
bash abx.sh -c 2 "yours|2300 1300 -50 294" "probe|2500 1400 0 294"    # interleaved A/B + mean W
```

Two rules: **concurrency is 1**, so no other chat may be in flight while probing, and always
**alternate** configs — the expert cache and prompt cache warm up over the first minutes, which
looks exactly like a clock gain (that mistake cost ~15 tok/s of phantom improvement).

