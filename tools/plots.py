# plots.py — draw the README charts (docs/img/*.png) from the measured numbers in README.md; edit DATA here when a result changes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs/img"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlelocation": "left",
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 14,
})


def finish(fig, ax, name, note):
    ax.text(0, -0.16, note, transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def hbars(name, title, rows, xmax, note, unit="tok/s", fmt="{:.0f}"):
    """rows: (label, value, highlighted, value_text or None)"""
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.6))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [BLUE if r[2] else GRAY for r in rows][::-1]
    bars = ax.barh(labels, vals, color=cols, height=0.62, edgecolor=SURF, linewidth=2)
    for b, r in zip(bars, rows[::-1]):
        ax.text(b.get_width() + xmax * 0.01, b.get_y() + b.get_height() / 2, r[3] or fmt.format(r[1]),
                va="center", fontsize=10, color=INK if r[2] else INK2, fontweight="bold" if r[2] else "normal")
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(unit)
    ax.set_title(title)
    finish(fig, ax, name, note)


def generation():
    hbars("gen-8k.png", "Generation speed, 8K context", [
        ("llama.cpp Vulkan", 13, False, None),
        ("llama.cpp ROCm", 25, False, None),
        ("Strata 0.1.62 (stock)", 78, False, "71–85"),
        ("Strata-rdna3 (fork's config)", 103.5, False, "103–104"),
        ("fork author's box (9950X3D2)", 116.5, False, "114–119"),
        ("this repo, daily 294 W", 129.6, True, "129.6"),
    ], 140, "RX 7900 XTX 24 GB · Ryzen 9 7945HX · 92 GB DDR5-5200 · Qwen3.8-Flash-Next IQ3_S (78 GB) · "
             "256 tokens, greedy")


def ladder():
    hbars("ladder.png", "What each step added (generation, 8K)", [
        ("fork's shipped config", 103.5, False, "103.5"),
        ("+ tuning, FAST=1 (skip low-ranked missed experts)", 110, False, "110"),
        ("+ kernel patches (VRAM 1300)", 113.5, False, "113.5"),
        ("+ VRAM 1400", 117.7, False, "117.7"),
        ("+ MTP draft-head work (q4_0, per-stream norm, retrained)", 129.6, True, "129.6"),
    ], 140, "Same box and model; each row includes the rows above it")


def chats():
    hbars("chats.png", "Varied chats with the MTP draft-head work", [
        ("shipped q2_0 draft head", 131.1, False, "131.1"),
        ("+ q4_0 draft experts (mtp-native)", 132.6, False, "132.6"),
        ("+ per-stream hidden norm (--mtp-hnorm stream)", 136.9, False, "136.9"),
        ("+ draft layer retrained on this model", 137.2, True, "137.2"),
    ], 160, "tools/chatbench.py, varied prompts, ≥3 alternated rounds (sweeps 42–44); a different bench from the 8K chart")


def prompt():
    rows = [("Strata 0.1.62 (stock)", 1900, 2000), ("Strata-rdna3 (fork's config)", 2641, 3266),
            ("this repo", 2903, 3605)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = range(len(rows))[::-1]
    h = 0.34
    for i, (lab, col) in enumerate([("8K prompt", BLUE), ("32K prompt", ORANGE)]):
        ys = [v + (h / 2 + 0.02) * (1 if i == 0 else -1) for v in y]
        vals = [r[1 + i] for r in rows]
        ax.barh(ys, vals, height=h, color=col, edgecolor=SURF, linewidth=2, label=lab)
        for yy, v in zip(ys, vals):
            ax.text(v + 40, yy, f"{v:,}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(0, 4300)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("tok/s")
    ax.set_title("Prompt reading")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    finish(fig, ax, "prompt.png", "~ for stock Strata: rounded in the original log")


def steptime():
    hbars("step-time.png", "GPU time per generation step (21.7 ms profiled, ~2.1 tokens kept)", [
        ("attention / linear-attention layers", 7.6, True, "7.6 ms"),
        ("experts in VRAM", 4.1, True, "4.1 ms"),
        ("hyper-connections", 3.4, True, "3.4 ms"),
        ("router + shared expert", 2.8, True, "2.8 ms"),
        ("copy + combine (waits on CPU experts)", 2.0, True, "2.0 ms"),
        ("waits (CPU, copies, gaps)", 1.1, True, "1.1 ms"),
        ("output head", 0.7, True, "0.7 ms"),
    ], 9, "8K context, STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 (profiling adds ~3 ms over the 18.5 ms unprofiled step)",
        unit="ms")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (generation, ladder, chats, prompt, steptime):
        f()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))
