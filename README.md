# start-strata.sh

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
