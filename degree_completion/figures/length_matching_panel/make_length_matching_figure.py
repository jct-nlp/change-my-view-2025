"""
Generates the length-matching figure in two modes, with different layouts:

    python make_length_matching_figure.py                     # presentation only (default)
    python make_length_matching_figure.py --mode report       # report version (today's
                                                                # originally-reviewed design)
    python make_length_matching_figure.py --mode all          # both

    Panel A — comment word-count distributions (Delta-awarded vs. not), before and
              after length matching. In both modes.
    Panel B — incremental PR-AUC gained by adding the 8 rhetorical-strategy scores,
              full scored population vs. length-matched sample. Report mode only --
              the presentation layout drops it; those numbers go in a small table
              placed directly on the slide instead, next to this now Panel-A-only image.

Both panels (where both exist) describe the SAME population and the SAME matching
procedure used in `this work/06_strategy_ablation.ipynb` (cells 2, 32, 34-36): the
2,753-row scored subset of the dataset (`results/05_cmv_comments_df.csv.zip`,
comments the LLM strategy-scoring pipeline in `05_rhetorical_scoring.ipynb` actually
scored), decile-bin matched down to 1,235 rows. This is NOT the same length-matched
sample as `04.3_length_controlled_analysis.ipynb` (that one is built from the full
3,971-row dataset and isn't restricted to scored comments) -- see README.md in this
folder for the four PR-AUC numbers (+0.0664 / +0.0557, 84% retained) if building the
slide's table from this script's output rather than typing them in directly.

Panel A's "after matching" distribution is produced by literally re-running 06's own
matching code (cell 32) against the saved results file below -- not re-derived or
approximated. Report mode's Panel B PR-AUC numbers are copied directly from 06's cell
35 output (length_benchmark_df) and are not recomputed here (that would require
retraining models, which this script deliberately does not do).

Presentation vs. report differences (the only place the two modes diverge):
  - layout:      report keeps Panel B (gain bar chart) beside Panel A, 16:9, stacked
                 Before/After; presentation drops Panel B entirely and lays Before/
                 After side by side instead -- see make_presentation_figure()'s
                 docstring for how its canvas width is derived to preserve Panel A's
                 exact absolute on-page size from the old 3-panel version.
  - header:      report keeps the figure-level title/subtitle; presentation drops
                 both to reclaim vertical space for a 15-minute talk, projected.
  - r annotation: report keeps the point-biserial r value per histogram; presentation
                 drops it (kept essential: distributions, means, n, takeaway).
  - footer:      report keeps full notebook paths/cell references/verification date;
                 presentation trims to a one-line source note.
  - scale:       presentation uses larger fonts throughout (axis labels, legends,
                 numeric annotations) for legibility when projected.
The word-count decile-bin matching note (Panel A) is shown in BOTH modes.

Run from anywhere:
    source ../../../.venv/bin/activate   # repo venv has matplotlib + pandas + scipy
    python make_length_matching_figure.py --mode all
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ---------------------------------------------------------------------------
# Palette -- copied from degree_completion/figures/make_figures.py so this
# figure matches the rest of the presentation's existing figures exactly.
# ---------------------------------------------------------------------------
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#FFFDF8"
DPI = 300

# ---------------------------------------------------------------------------
# Mode config -- the only place presentation/report diverge. See module
# docstring above for the rationale behind each difference.
# ---------------------------------------------------------------------------
MODE_CONFIG = {
    "presentation": dict(
        show_header=False,
        show_r_annotation=False,
        footer_style="minimal",
        scale=1.25,
        gs=dict(left=0.035, right=0.975, top=0.9, bottom=0.26, wspace=0.26),
        outdir=HERE,
    ),
    "report": dict(
        show_header=True,
        show_r_annotation=True,
        footer_style="full",
        scale=1.0,
        gs=dict(left=0.055, right=0.92, top=0.72, bottom=0.17, hspace=0.62, wspace=0.26),
        outdir=os.path.join(HERE, "report"),
    ),
}

MODE = "presentation"


def _cfg():
    return MODE_CONFIG[MODE]


def S(size):
    """Scale a font size for the current mode (presentation uses larger type for
    projection; report keeps the originally-reviewed sizes unchanged)."""
    return round(size * _cfg()["scale"], 1)


def _set_mode(mode):
    global MODE
    MODE = mode
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": S(12),
        "text.color": INK_PRIMARY,
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_PRIMARY,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    os.makedirs(_cfg()["outdir"], exist_ok=True)


# ---------------------------------------------------------------------------
# Data: reproduce 06_strategy_ablation.ipynb's own length-matched sample
# (cell 32) from the saved, already-scored results file. Read-only -- no
# model is fit or retrained anywhere in this script.
# ---------------------------------------------------------------------------
def load_scored_population_and_match():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "05_cmv_comments_df.csv.zip"))

    conv_df = df[df["is_convincing"] == 1].copy()
    nonconv_df = df[df["is_convincing"] == 0].copy()

    n_bins = 10
    bin_edges = pd.qcut(conv_df["word_count"], n_bins, duplicates="drop", retbins=True)[1]
    conv_df["length_bin"] = pd.cut(conv_df["word_count"], bins=bin_edges, include_lowest=True)
    nonconv_df["length_bin"] = pd.cut(nonconv_df["word_count"], bins=bin_edges, include_lowest=True)

    matched_parts = []
    for bin_label, group in conv_df.groupby("length_bin", observed=True):
        pool = nonconv_df[nonconv_df["length_bin"] == bin_label]
        take_n = min(len(group), len(pool))
        if take_n > 0:
            matched_parts.append(pool.sample(n=take_n, random_state=42))

    matched_nonconv_df = pd.concat(matched_parts) if matched_parts else nonconv_df.iloc[:0]
    matched_df = pd.concat([conv_df, matched_nonconv_df]).drop(columns="length_bin").sort_index()
    return df, matched_df


def _style_hist_axis(ax, show_xlabel, show_ticks=None):
    """show_xlabel controls the axis LABEL TEXT ("Comment word count..."); show_ticks
    controls the tick numbers independently (defaults to following show_xlabel, which is
    the report layout's behavior: the stacked top panel hides both together). The 1x3
    presentation layout needs tick numbers on both histograms but the label text only
    once, so it passes show_ticks explicitly."""
    if show_ticks is None:
        show_ticks = show_xlabel
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=S(10.5))
    ax.set_xlim(0, 600)
    if show_xlabel:
        ax.set_xlabel("Comment word count (trimmed at 600 words)", fontsize=S(11),
                       color=INK_SECONDARY, labelpad=6)
    if not show_ticks:
        ax.set_xticklabels([])


def _plot_length_panel(ax, sample_df, title, show_xlabel, show_legend, show_ticks=None):
    trimmed = sample_df[sample_df["word_count"] < 600]
    bins = range(0, 650, 50)

    no_delta = trimmed.loc[trimmed["is_convincing"] == 0, "word_count"]
    delta = trimmed.loc[trimmed["is_convincing"] == 1, "word_count"]
    n_no, n_yes, n_total = (sample_df["is_convincing"] == 0).sum(), (sample_df["is_convincing"] == 1).sum(), len(sample_df)
    mean_no, mean_yes = sample_df.loc[sample_df["is_convincing"] == 0, "word_count"].mean(), \
                         sample_df.loc[sample_df["is_convincing"] == 1, "word_count"].mean()

    ax.hist(no_delta, bins=bins, density=True, color=COLOR_BLUE, alpha=0.55, zorder=3,
            label=f"No Δ (n={n_no:,}, {n_no/n_total*100:.0f}%)")
    ax.hist(delta, bins=bins, density=True, color=COLOR_ORANGE, alpha=0.55, zorder=4,
            label=f"Δ Awarded (n={n_yes:,}, {n_yes/n_total*100:.0f}%)")

    ax.axvline(mean_no, color=COLOR_BLUE, linestyle=(0, (5, 3)), linewidth=1.5, zorder=5)
    ax.axvline(mean_yes, color=COLOR_ORANGE, linestyle=(0, (5, 3)), linewidth=1.5, zorder=5)
    ax.annotate(f"mean {mean_no:.0f}", xy=(mean_no, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-5, -3), textcoords="offset points", va="top", ha="right",
                fontsize=S(11), color=COLOR_BLUE, fontweight="bold")
    ax.annotate(f"mean {mean_yes:.0f}", xy=(mean_yes, 1.0), xycoords=("data", "axes fraction"),
                xytext=(5, -3), textcoords="offset points", va="top", ha="left",
                fontsize=S(11), color=COLOR_ORANGE, fontweight="bold")

    ax.set_title(title, fontsize=S(14), color=INK_PRIMARY, loc="left", fontweight="bold", pad=10)

    if _cfg()["show_r_annotation"]:
        from scipy.stats import pointbiserialr
        r, _ = pointbiserialr(sample_df["is_convincing"], sample_df["word_count"])
        ax.text(1.0, 1.02, f"point-biserial r = {r:.3f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=S(10), color=INK_MUTED, style="italic")

    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=S(11.5), bbox_to_anchor=(1.0, 0.98))

    _style_hist_axis(ax, show_xlabel, show_ticks=show_ticks)


def _plot_gain_panel(ax, labels_inside=False):
    """labels_inside=False (report, unchanged): sample descriptions as y-axis tick labels
    to the left of the bars. labels_inside=True (presentation): same descriptions printed
    inside each bar instead, freeing the left margin so this panel can take less width."""
    gains = [0.0664, 0.0557]
    colors = [COLOR_BLUE, COLOR_ORANGE]
    retention_pct = gains[1] / gains[0] * 100  # 83.9%, reported as "84%"

    bar_height = 0.62 if labels_inside else 0.5
    y_pos = range(len(gains))
    bars = ax.barh(y_pos, gains, color=colors, height=bar_height, zorder=3)
    ax.set_ylim(-0.85, 1.85)

    for bar, val in zip(bars, gains):
        ax.text(val + 0.0015, bar.get_y() + bar.get_height() / 2, f"+{val:.4f} PR-AUC",
                va="center", ha="left", fontsize=S(15), color=INK_PRIMARY, fontweight="bold")

    bracket_x = 0.070
    ax.plot([bracket_x, bracket_x], [0, 1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.004, bracket_x], [0, 0], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.004, bracket_x], [1, 1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.text(bracket_x + 0.004, 0.5, f"{retention_pct:.0f}% retained",
            va="bottom", ha="left", fontsize=S(15), color=INK_PRIMARY, fontweight="bold", clip_on=False)
    ax.text(bracket_x + 0.004, 0.5, "descriptive estimate,\nnot a significance test",
            va="top", ha="left", fontsize=S(9), color=INK_MUTED, style="italic", clip_on=False, linespacing=1.3)

    if labels_inside:
        in_bar_main = ["Full population", "Length matched"]
        in_bar_n = ["n = 2,753", "n = 1,235"]
        ax.set_yticks([])
        for bar, main, nlabel in zip(bars, in_bar_main, in_bar_n):
            cy = bar.get_y() + bar.get_height() / 2
            ax.text(0.003, cy + 0.12, main, va="center", ha="left",
                    fontsize=S(12.5), color="white", fontweight="bold")
            ax.text(0.003, cy - 0.12, nlabel, va="center", ha="left",
                    fontsize=S(10), color="white", fontweight="normal")
    else:
        labels = ["Full scored\npopulation (n=2,753)", "Length-matched\nsample (n=1,235)"]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=S(14))
        ax.tick_params(axis="y", length=0, labelsize=S(14))

    ax.invert_yaxis()
    ax.set_xlim(0, 0.125)
    ax.set_title("Strategy-score PR-AUC gain", fontsize=S(14), color=INK_PRIMARY, loc="left",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Incremental PR-AUC", fontsize=S(12), color=INK_SECONDARY, labelpad=8)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(axis="x", length=0, labelsize=S(11))


def _matching_note_text():
    return "Matched by word-count decile bin, not exact-length pairs"


def _caveat_text():
    return "Descriptive comparison across different samples; not proof of independence from length."


def make_report_figure():
    """2x2 stacked layout (Before/After stacked left, gain panel spanning right), 16:9 --
    today's originally-reviewed report design, unchanged."""
    full_df, matched_df = load_scored_population_and_match()
    cfg = _cfg()
    gs_kw = cfg["gs"]

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1],
                            hspace=gs_kw["hspace"], wspace=gs_kw["wspace"],
                            left=gs_kw["left"], right=gs_kw["right"],
                            top=gs_kw["top"], bottom=gs_kw["bottom"])

    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[1, 0], sharex=ax_before)
    ax_gain = fig.add_subplot(gs[:, 1])

    _plot_length_panel(ax_before, full_df, "Before matching", show_xlabel=False, show_legend=True)
    _plot_length_panel(ax_after, matched_df, "After matching", show_xlabel=True, show_legend=False)
    _plot_gain_panel(ax_gain)

    ax_before.annotate("A", xy=(0, 1.0), xycoords="axes fraction", xytext=(-4, 48),
                        textcoords="offset points", fontsize=S(20), fontweight="bold", color=INK_PRIMARY)
    ax_gain.annotate("B", xy=(0, 1.0), xycoords="axes fraction", xytext=(-4, 48),
                      textcoords="offset points", fontsize=S(20), fontweight="bold", color=INK_PRIMARY)

    fig.suptitle("Rhetorical-Strategy Signal Survives Length Matching",
                 x=0.045, y=0.965, ha="left", va="top", fontsize=S(19), fontweight="bold",
                 color=INK_PRIMARY)
    fig.text(0.045, 0.905,
             "Length matching narrows the comment-length gap between outcome classes — most of the\n"
             "rhetorical-strategy scores' predictive gain survives it anyway",
             ha="left", va="top", fontsize=S(12.5), color=INK_SECONDARY, linespacing=1.35)

    caveat_y = 0.055
    fig.text(0.5, caveat_y, _caveat_text(), ha="center", va="bottom", fontsize=S(10.5),
             color=INK_MUTED, style="italic")

    note_y = caveat_y + 0.048
    fig.text(0.045, note_y, _matching_note_text(), ha="left", va="bottom", fontsize=S(9.5),
             color=INK_MUTED, style="italic")

    fig.text(0.045, 0.012,
             "Source: this work/06_strategy_ablation.ipynb, cells 2 (n=2,753 scored population), 32 (decile-bin length "
             "matching, random_state=42, n=1,235) and 35 (PR-AUC gain, headline_df/length_benchmark_df). Data: "
             "results/05_cmv_comments_df.csv.zip. Matched by word-count decile bin, not exact-length pairs; distinct "
             "from the (non-comparable) sample in 04.3_length_controlled_analysis.ipynb. Verified 2026-10-21.",
             ha="left", va="bottom", fontsize=S(7.6), color=INK_MUTED, linespacing=1.3, wrap=True)

    _save(fig)


def make_presentation_figure():
    """Left side only: Before | After histograms side by side, no gain panel -- the PR-AUC
    numbers go in a separate table placed directly on the slide instead. Canvas width is
    derived (not hand-picked) so each histogram panel keeps the exact same absolute size
    (same font scale, same subplot width in inches) it had inside the old 3-panel figure --
    only the canvas is cropped down to where Panel B used to be, nothing is re-stretched."""
    full_df, matched_df = load_scored_population_and_match()
    cfg = _cfg()
    gs_kw = cfg["gs"]

    # Old 3-column layout's panel width/gap/margins, in inches -- computed from a throwaway
    # figure rather than hand-derived, so the two histogram panels below end up PIXEL-FOR-
    # PIXEL the same absolute size they were inside the old 3-panel figure (same font scale,
    # same subplot width), not just the same *relative* proportions on a re-stretched canvas.
    # Panel B is gone (replaced by a table on the slide itself), so only the canvas shrinks.
    OLD_FIGSIZE_W = 16.0
    _dummy_fig = plt.figure(figsize=(OLD_FIGSIZE_W, 6.8))
    _dummy_gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 0.9], wspace=gs_kw["wspace"],
                                   left=gs_kw["left"], right=gs_kw["right"])
    _p0 = _dummy_fig.add_subplot(_dummy_gs[0, 0]).get_position()
    _p1 = _dummy_fig.add_subplot(_dummy_gs[0, 1]).get_position()
    plt.close(_dummy_fig)
    left_margin_in = _p0.x0 * OLD_FIGSIZE_W
    panel_w_in = _p0.width * OLD_FIGSIZE_W
    gap_in = (_p1.x0 - _p0.x1) * OLD_FIGSIZE_W
    right_margin_in = left_margin_in  # symmetric margin, same absolute size as the left one

    fig_height = 6.8
    fig_width = left_margin_in + 2 * panel_w_in + gap_in + right_margin_in
    wspace_new = gap_in / panel_w_in  # closed form for 2 equal-width columns

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=wspace_new,
                            left=left_margin_in / fig_width, right=1 - right_margin_in / fig_width,
                            top=gs_kw["top"], bottom=gs_kw["bottom"])

    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[0, 1], sharex=ax_before, sharey=ax_before)

    # Tick numbers shown on both histograms (each needs its own readable scale), but the
    # "Comment word count..." label text only once, centered beneath the pair below --
    # no panel letters (A/B): removed to reclaim the top margin for taller panels. Legend
    # suppressed on both axes here -- added once below, as a figure-level legend centered
    # in the gap between the two panels, at the same height it sat at originally.
    _plot_length_panel(ax_before, full_df, "Before matching", show_xlabel=False, show_legend=False, show_ticks=True)
    _plot_length_panel(ax_after, matched_df, "After matching", show_xlabel=False, show_legend=False, show_ticks=True)

    pos_before = ax_before.get_position()
    pos_after = ax_after.get_position()
    shared_xlabel_x = (pos_before.x0 + pos_after.x1) / 2
    fig.text(shared_xlabel_x, pos_before.y0 - 0.035, "Comment word count (trimmed at 600 words)",
             ha="center", va="top", fontsize=S(12), color=INK_SECONDARY)

    # Legend: horizontally centered in the gap between the two histograms (not inside
    # either one), nudged slightly left of true-center and down a touch from the
    # "mean"-label row height, per visual feedback that true-center read as too far right.
    handles, labels = ax_before.get_legend_handles_labels()
    gap_center_x = (pos_before.x1 + pos_after.x0) / 2
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(gap_center_x - 0.02, pos_before.y1 - 0.035),
               bbox_transform=fig.transFigure, frameon=False, fontsize=S(11.5))

    # Caveat line vertically centered between the x-axis label and the source line below,
    # instead of sitting right on top of the source line.
    xlabel_y = pos_before.y0 - 0.035
    source_y = 0.008
    caveat_y = (xlabel_y + source_y) / 2
    fig.text(0.5, caveat_y, _caveat_text(), ha="center", va="center", fontsize=S(13.5),
             color=INK_MUTED, style="italic")

    # Matching-method note moved INTO the "After matching" panel's own open space (bars
    # taper off past ~300 words, leaving room upper-right), larger font, two lines, rather
    # than sitting below the axis.
    ax_after.text(0.97, 0.58, "Matched by word-count\ndecile bin, not\nexact-length pairs",
                  transform=ax_after.transAxes, ha="right", va="center", fontsize=S(11.5),
                  color=INK_MUTED, style="italic", linespacing=1.4)

    fig.text(0.03, 0.008, "Source: 06_strategy_ablation.ipynb · results/05_cmv_comments_df.csv.zip",
             ha="left", va="bottom", fontsize=S(9), color=INK_MUTED, linespacing=1.3)

    _save(fig)


def _save(fig):
    outdir = _cfg()["outdir"]
    out_png = os.path.join(outdir, "length_matching_strategy_gain.png")
    out_svg = os.path.join(outdir, "length_matching_strategy_gain.svg")
    fig.savefig(out_png, dpi=DPI, facecolor=SURFACE)
    fig.savefig(out_svg, facecolor=SURFACE)
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["presentation", "report", "all"], default="presentation",
                         help="presentation: slide-ready, no header, larger fonts, minimal footer "
                              "(default). report: today's originally-reviewed design (title/subtitle, "
                              "point-biserial r, full footer) -> report/ subfolder. all: both.")
    args = parser.parse_args()
    modes = ["presentation", "report"] if args.mode == "all" else [args.mode]
    for mode in modes:
        _set_mode(mode)
        if mode == "presentation":
            make_presentation_figure()
        else:
            make_report_figure()
