"""
Generates the first three MSc-defense presentation figures from verified notebook outputs.

Every numeric constant below was read directly from the cited notebook cell (see
FIGURE_NOTES.md for the full source trail) -- nothing here is estimated or re-derived.
Run from this directory (or anywhere, paths are relative to this file):

    source ../../.venv/bin/activate   # repo venv has matplotlib
    python make_figures.py

Outputs PNG (300 dpi) and SVG for each figure into this directory.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared style -- one palette, one typography system, one figure size, reused
# across all three figures (dataviz skill's validated default categorical
# palette, first three slots: blue / orange / aqua).
# ---------------------------------------------------------------------------
COLOR_BLUE = "#2a78d6"    # reference / baseline bars
COLOR_ORANGE = "#eb6834"  # the headline / highlighted finding
COLOR_AQUA = "#1baf7a"    # third supporting category
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#898781"
SURFACE = "#fcfcfb"

FIGSIZE = (9.5, 5.2)
DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style_axes(ax, xlabel, xlim):
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=12, color=INK_SECONDARY, labelpad=10)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(axis="y", length=0, labelsize=13)
    ax.tick_params(axis="x", length=0, labelsize=11)


def _save(fig, name):
    png_path = os.path.join(HERE, f"{name}.png")
    svg_path = os.path.join(HERE, f"{name}.svg")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


# ---------------------------------------------------------------------------
# Figure 1 -- Initial model comparison
# Source: this work/04.2_baseline_comparison.ipynb, cells 9-10, 33
# ---------------------------------------------------------------------------
def figure1():
    labels = [
        "Selected engineered-feature\npipeline (XGBoost + SMOTE,\n229 features)",
        "TF-IDF + Logistic\nRegression",
        "Word count alone\n(1 feature)",
    ]
    values = [0.3848, 0.5119, 0.5256]
    colors = [COLOR_BLUE, COLOR_AQUA, COLOR_ORANGE]
    no_skill = 0.2031  # majority-class PR-AUC = test-fold positive rate (n=709, 81 threads)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", ha="left", fontsize=13, color=INK_PRIMARY, fontweight="bold")

    ax.axvline(no_skill, color=BASELINE_LINE, linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
    ax.annotate(f"No-skill baseline\n(majority class) = {no_skill:.3f}",
                xy=(no_skill, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -4), textcoords="offset points",
                va="top", ha="left", fontsize=9.5, color=INK_MUTED, style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.invert_yaxis()
    _style_axes(ax, "Held-out PR-AUC", (0, 0.62))

    fig.suptitle("Word Count Alone Outperforms the Engineered-Feature Pipeline",
                 x=0.06, y=0.97, ha="left", fontsize=15.5, color=INK_PRIMARY, fontweight="bold")
    fig.text(0.06, 0.885, "Held-out test set: 709 comments, 81 threads (thread-disjoint split) — same test fold for all three models",
              fontsize=10.5, color=INK_SECONDARY)

    fig.text(0.06, 0.015,
              "Source: this work/04.2_baseline_comparison.ipynb, cells 9–10 (engineered pipeline, TF-IDF) and 33 (word count). "
              "PR-AUC values and no-skill baseline verified 2026-10-14.",
              fontsize=8.5, color=INK_MUTED)

    fig.subplots_adjust(left=0.30, right=0.95, top=0.78, bottom=0.16)
    _save(fig, "figure1_initial_model_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- Incremental value of rhetorical strategies
# Source: this work/06_strategy_ablation.ipynb, cell 17
# ---------------------------------------------------------------------------
def figure2():
    labels = [
        "Engineered-feature model\n(no rhetorical scores)",
        "Engineered-feature model\n+ 8 rhetorical-strategy scores",
        "Rhetorical-strategy\nscores only",
    ]
    means = [0.4467, 0.5202, 0.3538]
    stds = [0.0196, 0.0177, 0.0079]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA]
    no_skill = 0.2382  # test-fold positive rate, scored subset (n=2,753; test 592/80 threads)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, height=0.55, zorder=3,
                    error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.6, capsize=5, capthick=1.6))

    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.012, bar.get_y() + bar.get_height() / 2, f"{mean:.4f} ± {std:.4f}",
                va="center", ha="left", fontsize=12.5, color=INK_PRIMARY, fontweight="bold")

    ax.axvline(no_skill, color=BASELINE_LINE, linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
    ax.annotate(f"No-skill baseline\n(scored subset) = {no_skill:.3f}",
                xy=(no_skill, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -4), textcoords="offset points",
                va="top", ha="left", fontsize=9.5, color=INK_MUTED, style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.invert_yaxis()
    _style_axes(ax, "Held-out PR-AUC (mean ± SD across 8 column orderings)", (0, 0.68))

    fig.suptitle("Rhetorical-Strategy Scores Add Predictive Value Beyond Engineered Features",
                 x=0.06, y=0.97, ha="left", fontsize=14.5, color=INK_PRIMARY, fontweight="bold")
    fig.text(0.06, 0.875,
              "Scored subset (n=2,753; test: 592 comments, 80 threads) — error bars are ± SD across 8 random\n"
              "feature-column orderings of the same fixed XGBoost+SMOTE architecture, NOT bootstrap confidence intervals",
              fontsize=10, color=INK_SECONDARY)

    fig.text(0.06, 0.015,
              "Source: this work/06_strategy_ablation.ipynb, cell 17 (headline_df). Values verified 2026-10-14. "
              "Column-order sensitivity documented in cells 11–14 of the same notebook.",
              fontsize=8.5, color=INK_MUTED)

    fig.subplots_adjust(left=0.32, right=0.95, top=0.76, bottom=0.16)
    _save(fig, "figure2_strategy_incremental_value")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- Rhetorical-strategy gain under length matching
# Source: this work/06_strategy_ablation.ipynb, cells 32 and 35
# ---------------------------------------------------------------------------
def figure3():
    labels = [
        "Full scored population\n(n = 2,753)",
        "Length-matched sample\n(within scored population, n = 1,235)",
    ]
    gains = [0.0664, 0.0557]
    colors = [COLOR_BLUE, COLOR_ORANGE]
    retention_pct = 0.0557 / 0.0664 * 100  # 83.9%, reported as "84%"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, gains, color=colors, height=0.42, zorder=3)

    for bar, val in zip(bars, gains):
        ax.text(val + 0.0015, bar.get_y() + bar.get_height() / 2, f"+{val:.4f} PR-AUC",
                va="center", ha="left", fontsize=13, color=INK_PRIMARY, fontweight="bold")

    # Bracket + retention callout between the two bars, kept well inside xlim so it
    # can't inflate the saved bbox past the footer text's wrap width.
    y0, y1 = 0, 1
    bracket_x = 0.080
    ax.plot([bracket_x, bracket_x], [y0, y1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.004, bracket_x], [y0, y0], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.004, bracket_x], [y1, y1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.text(bracket_x + 0.004, (y0 + y1) / 2, f"{retention_pct:.0f}% of the gain\nis retained",
            va="center", ha="left", fontsize=12.5, color=INK_PRIMARY, fontweight="bold", clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.invert_yaxis()
    _style_axes(ax, "Incremental PR-AUC gain from adding 8 rhetorical-strategy scores", (0, 0.10))

    fig.suptitle("The Rhetorical-Strategy Gain Survives Length Matching",
                 x=0.06, y=0.96, ha="left", va="top", fontsize=15.5, color=INK_PRIMARY, fontweight="bold")
    fig.text(0.06, 0.78,
              "Same engineered-feature baseline (‘trusted’ columns), same scored-subset population, before vs. after a strict\n"
              "1:1 length match built within it — this is a comparison of incremental PR-AUC gain, not of raw PR-AUC across\n"
              "populations with different class prevalence (23.8% vs. 51.3% positive)",
              va="top", fontsize=9.8, color=INK_SECONDARY, linespacing=1.5)

    fig.text(0.06, 0.015,
              "Source: this work/06_strategy_ablation.ipynb, cells 32 (matching) and 35 (length_benchmark_df). Values verified 2026-10-14.\n"
              "This length-matched sample is constructed within the 2,753-row scored population and is a separate, differently-built sample\n"
              "from 04.3_length_controlled_analysis.ipynb's own length-matched sample (built from the full 3,971-row dataset); the two are not\n"
              "combined here.",
              va="bottom", fontsize=8.2, color=INK_MUTED, linespacing=1.4)

    fig.subplots_adjust(left=0.30, right=0.92, top=0.62, bottom=0.26)
    _save(fig, "figure3_strategy_gain_length_matching")
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print("\nAll three figures generated.")
