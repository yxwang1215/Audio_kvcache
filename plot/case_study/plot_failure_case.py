#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACL-style case study figure: SnapKV repetition failure vs AudioKV stable output.
LibriSpeech Chapter 13754, duration 232.62 s.
Outputs: failure_case_visualization.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── ACL global rcParams ────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          9.5,
        "axes.unicode_minus": False,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    }
)

# ── ACL palette: warm sienna-amber (failure) vs steel blue (success) ──────────
C_SNAP_FG   = "#a0522d"   # SnapKV label / WER  – sienna
C_SNAP_BG   = "#fef9f4"   # SnapKV panel fill   – warm cream
C_SNAP_EDGE = "#c87941"   # SnapKV border       – terra-cotta amber
C_SNAP_LOOP = "#b85c28"   # repetition text     – deep amber
C_SNAP_ONS  = "#7d3410"   # failure-onset phrase – dark burnt sienna

C_AKV_FG    = "#1f5e99"   # AudioKV label / WER – deep steel blue
C_AKV_BG    = "#f4f8fd"   # AudioKV panel fill  – light sky
C_AKV_EDGE  = "#4a86c8"   # AudioKV border      – steel blue

C_BODY      = "#1c1c1c"   # normal transcript body
C_DIM       = "#888888"   # de-emphasised text
C_HDR_BG    = "#f0f0f0"   # header band fill
C_HDR_FG    = "#2e2e2e"   # header text
C_DIV       = "#c8c8c8"   # panel separator

# Shared transcript prefix (truncated for space, ≤46 chars/line)
PREFIX_LINES = [
    "They knew no north, no south…  When",
    "Brigham Young passed from earth, he was",
    "mourned as Moses of Israel.  He had led",
    "the people into the land of holy promise,",
    "and divided unto them their inheritances.",
]

SNAP_ONSET = '"…future states shall recognize"'
SNAP_LOOP  = "every future state shall recognize"

AKV_CONT_LINES = [
    '"…be recognized every sanctified and',
    'authorized relationship existing here,',
    'of parent and child, brother and sister,',
    'husband and wife.  Contracts of marriage',
    'must be sanctioned by a power greater',
    'than that of earth."',
]


def _bg_rect(ax, facecolor, edgecolor, lw=1.5):
    """Draw a full-axes background rectangle (no rounding – cleaner for ACL)."""
    rect = mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="square,pad=0.0",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)


def _t(ax, x, y, s, **kw):
    """Shorthand for ax.text in axes-fraction coordinates."""
    return ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left",
                   clip_on=True, **kw)


def _divider(ax, y, color=C_DIV, lw=0.7, ls="--"):
    ax.axhline(y=y, xmin=0.03, xmax=0.97, color=color, linewidth=lw,
               linestyle=ls, zorder=2)


def draw_snap_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    _bg_rect(ax, C_SNAP_BG, C_SNAP_EDGE)

    # ── header ────────────────────────────────────────────────────────────────
    _t(ax, 0.045, 0.963, "SnapKV",
       fontsize=11, fontweight="bold", color=C_SNAP_FG)
    ax.text(0.955, 0.963, "WER = 93.62 %",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=11.5, fontweight="bold", color=C_SNAP_FG, clip_on=False)
    ax.axhline(y=0.920, xmin=0.0, xmax=1.0,
               color=C_SNAP_EDGE, linewidth=1.0, linestyle="-")

    # ── shared prefix ─────────────────────────────────────────────────────────
    y = 0.888
    DY = 0.073
    for line in PREFIX_LINES:
        _t(ax, 0.045, y, line, fontsize=8.8, color=C_BODY)
        y -= DY

    # ── failure-onset marker ──────────────────────────────────────────────────
    y -= 0.016
    _divider(ax, y + 0.012, color=C_SNAP_EDGE, lw=0.6, ls=":")
    y -= 0.008
    _t(ax, 0.045, y, "▸ Failure onset",
       fontsize=8.5, fontweight="bold", color=C_SNAP_ONS)
    y -= DY
    _t(ax, 0.055, y, SNAP_ONSET,
       fontsize=8.8, fontstyle="italic", color=C_SNAP_ONS)
    y -= DY + 0.008

    # ── repetition loop ───────────────────────────────────────────────────────
    for _ in range(4):
        _t(ax, 0.055, y, SNAP_LOOP,
           fontsize=8.8, fontstyle="italic", color=C_SNAP_LOOP)
        y -= DY
    _t(ax, 0.055, y, "[ repeats ×hundreds … ]",
       fontsize=8.2, color=C_DIM, fontstyle="italic")


def draw_akv_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    _bg_rect(ax, C_AKV_BG, C_AKV_EDGE)

    # ── header ────────────────────────────────────────────────────────────────
    _t(ax, 0.045, 0.963, "AudioKV",
       fontsize=11, fontweight="bold", color=C_AKV_FG)
    ax.text(0.955, 0.963, "WER = 1.09 %",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=11.5, fontweight="bold", color=C_AKV_FG, clip_on=False)
    ax.axhline(y=0.920, xmin=0.0, xmax=1.0,
               color=C_AKV_EDGE, linewidth=1.0, linestyle="-")

    # ── shared prefix ─────────────────────────────────────────────────────────
    y = 0.888
    DY = 0.073
    for line in PREFIX_LINES:
        _t(ax, 0.045, y, line, fontsize=8.8, color=C_BODY)
        y -= DY

    # ── coherent continuation marker ──────────────────────────────────────────
    y -= 0.016
    _divider(ax, y + 0.012, color=C_AKV_EDGE, lw=0.6, ls=":")
    y -= 0.008
    _t(ax, 0.045, y, "▸ Coherent continuation",
       fontsize=8.5, fontweight="bold", color=C_AKV_FG)
    y -= DY

    # ── correct output ────────────────────────────────────────────────────────
    for i, line in enumerate(AKV_CONT_LINES):
        _t(ax, 0.055, y, line,
           fontsize=8.8, fontstyle="italic",
           color=C_BODY if i < len(AKV_CONT_LINES) - 1 else C_AKV_FG)
        y -= DY
    y -= 0.008
    _t(ax, 0.055, y, "No repetition observed.",
       fontsize=8.5, fontweight="bold", color=C_AKV_FG)


def main():
    fig = plt.figure(figsize=(7.6, 4.55))

    # header row (thin) + main panels row
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[0.072, 1.0],
        hspace=0.0,
        wspace=0.035,
    )

    ax_hdr  = fig.add_subplot(gs[0, :])
    ax_snap = fig.add_subplot(gs[1, 0])
    ax_akv  = fig.add_subplot(gs[1, 1])

    # ── Header band ───────────────────────────────────────────────────────────
    ax_hdr.set_axis_off()
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="square,pad=0.0",
            facecolor=C_HDR_BG,
            edgecolor=C_DIV,
            linewidth=0.8,
            transform=ax_hdr.transAxes,
            clip_on=False,
        )
    )
    ax_hdr.text(
        0.5, 0.50,
        "Case Study  ·  LibriSpeech Ch. 13754  ·  Duration: 232.62 s  ·  KV budget: 512 tokens",
        transform=ax_hdr.transAxes,
        va="center", ha="center",
        fontsize=8.8, color=C_HDR_FG,
    )

    # ── Panels ────────────────────────────────────────────────────────────────
    draw_snap_panel(ax_snap)
    draw_akv_panel(ax_akv)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent
    pdf_path = output_dir / "failure_case_visualization.pdf"
    png_path = output_dir / "failure_case_visualization.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", bbox_inches="tight", facecolor="white")

    print(f"Saved PDF → {pdf_path}")
    print(f"Saved PNG → {png_path}")


if __name__ == "__main__":
    main()
