"""
Generates all seven MSc-defense presentation figures from verified notebook outputs.

Every numeric constant below was read directly from the cited notebook cell (see
FIGURE_NOTES.md for the full source trail, exact definitions, and caveats) -- nothing here
is estimated or re-derived except Figure 4's raw word_count array, which is recomputed from
the saved results CSV using the exact same one-line formula as 04.2_baseline_comparison.ipynb
cell 30 (read-only extraction, no model refit).

Run from this directory (or anywhere, paths are relative to this file):

    source ../../.venv/bin/activate   # repo venv has matplotlib + pandas
    python make_figures.py

Outputs PNG (300 dpi) and SVG for each figure into this directory. All seven figures share
one figure size (16:9, for direct use on a presentation slide), one typography system, one
color system, and one title/subtitle/footer layout -- see the shared helpers below.
"""

import os
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ---------------------------------------------------------------------------
# Shared style -- one palette, one typography system, one figure size, reused
# across all seven figures (dataviz skill's validated default categorical
# palette: blue / orange / aqua / yellow, first four slots).
# ---------------------------------------------------------------------------
COLOR_BLUE = "#2a78d6"      # reference / baseline / "population A" / "dataset A"
COLOR_ORANGE = "#eb6834"    # the headline / highlighted finding / "population B" / "dataset B"
COLOR_AQUA = "#1baf7a"      # third supporting category
COLOR_GRAY_LINE = "#c3c2b7"  # de-emphasized context series (bump chart, literature bands)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#898781"
SURFACE = "#FFFDF8"

FIGSIZE = (10, 5.625)  # 16:9 -- drops directly onto a presentation slide
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


# ---------------------------------------------------------------------------
# Shared helpers -- identical title/subtitle/footer anchoring on every figure.
# ---------------------------------------------------------------------------
def _header(fig, title, subtitle=None, title_size=17, show_title=True, subtitle_size=11,
            subtitle_center=False):
    """show_title=False omits the on-chart title (e.g. when the slide's own headline already
    states it) and moves the subtitle up into the freed space. subtitle_center=True centers the
    subtitle on the figure instead of left-aligning it at the standard margin."""
    if show_title:
        fig.suptitle(title, x=0.055, y=0.95, ha="left", va="top",
                     fontsize=title_size, fontweight="bold", color=INK_PRIMARY)
        subtitle_y = 0.855
    else:
        subtitle_y = 0.95
    if subtitle:
        if subtitle_center:
            fig.text(0.5, subtitle_y, subtitle, ha="center", va="top",
                      fontsize=subtitle_size, color=INK_SECONDARY, linespacing=1.4)
        else:
            fig.text(0.055, subtitle_y, subtitle, ha="left", va="top",
                      fontsize=subtitle_size, color=INK_SECONDARY, linespacing=1.4)


def _footer(fig, text, center=False):
    if center:
        fig.text(0.5, 0.015, text, ha="center", va="bottom",
                  fontsize=7.8, color=INK_MUTED, linespacing=1.35)
    else:
        fig.text(0.055, 0.015, text, ha="left", va="bottom",
                  fontsize=7.8, color=INK_MUTED, linespacing=1.35)


def _style_axes(ax, xlabel, xlim):
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=11, color=INK_SECONDARY, labelpad=8)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(axis="y", length=0, labelsize=12.5)
    ax.tick_params(axis="x", length=0, labelsize=10.5)


def _no_skill_line(ax, x, label, fontsize=9):
    ax.axvline(x, color=BASELINE_LINE, linestyle=(0, (5, 3)), linewidth=1.5, zorder=2)
    ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -3), textcoords="offset points",
                va="top", ha="left", fontsize=fontsize, color=INK_MUTED, style="italic")


def _save(fig, name, facecolor=None):
    """facecolor overrides the shared SURFACE color for this figure only (the rcParam
    savefig.facecolor is fixed to SURFACE, so a per-figure override must be passed explicitly
    here, not just set on the Figure/Axes)."""
    png_path = os.path.join(HERE, f"{name}.png")
    svg_path = os.path.join(HERE, f"{name}.svg")
    save_kwargs = {"bbox_inches": "tight"}
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    fig.savefig(png_path, dpi=DPI, **save_kwargs)
    fig.savefig(svg_path, **save_kwargs)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def _load_comment_word_counts():
    """Recomputes word_count the same way 04.2_baseline_comparison.ipynb cell 30 does, from
    the saved, already-cleaned results CSV. Read-only extraction -- no model fit, no rerun."""
    zpath = os.path.join(RESULTS_DIR, "03_cmv_comments_df.csv.zip")
    with zipfile.ZipFile(zpath) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df["word_count"] = df["cleaned_final_comment"].fillna("").apply(lambda s: len(s.split()))
    return df[["word_count", "is_convincing"]]


# ---------------------------------------------------------------------------
# Figure 1 -- Initial model comparison
# Source: this work/04.2_baseline_comparison.ipynb, cells 9-10, 33
# ---------------------------------------------------------------------------
def figure1():
    labels = ["Engineered pipeline\n(229 features)", "TF-IDF +\nLogistic Regression", "Word count alone\n(1 feature)"]
    values = [0.3848, 0.5119, 0.5256]
    colors = [COLOR_BLUE, COLOR_AQUA, COLOR_ORANGE]
    no_skill = 0.2031  # majority-class PR-AUC = test-fold positive rate

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", ha="left", fontsize=14, color=INK_PRIMARY, fontweight="bold")

    _no_skill_line(ax, no_skill, f"No-skill = {no_skill:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    _style_axes(ax, "Held-out PR-AUC", (0, 0.62))

    _header(fig, "Word Count Beats the Engineered Pipeline",
            "Held-out PR-AUC, same 709-comment test set for all three models",
            show_title=False, subtitle_size=15)
    _footer(fig, "Source: 04.2_baseline_comparison.ipynb, cells 9–10, 33.")

    fig.subplots_adjust(left=0.27, right=0.95, top=0.88, bottom=0.14)
    _save(fig, "figure1_initial_model_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- Incremental value of rhetorical strategies
# Source: this work/06_strategy_ablation.ipynb, cell 17
# ---------------------------------------------------------------------------
def figure2():
    labels = ["Engineered features\n(no strategy scores)", "Engineered features\n+ 8 strategy scores", "Strategy scores\nonly"]
    means = [0.4467, 0.5202, 0.3538]
    stds = [0.0196, 0.0177, 0.0079]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA]
    no_skill = 0.2382  # test-fold positive rate, scored subset

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, height=0.55, zorder=3,
                    error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.5, capsize=5, capthick=1.5))

    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.012, bar.get_y() + bar.get_height() / 2, f"{mean:.4f}",
                va="center", ha="left", fontsize=13.5, color=INK_PRIMARY, fontweight="bold")

    # Gain bracket between bar 0 and bar 1, same visual language as Figure 3's retention bracket.
    bracket_x = 0.60
    ax.plot([bracket_x, bracket_x], [0, 1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.01, bracket_x], [0, 0], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.01, bracket_x], [1, 1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.text(bracket_x + 0.012, 0.5, "+0.0735", va="center", ha="left", fontsize=14,
            color=INK_PRIMARY, fontweight="bold", clip_on=False)

    _no_skill_line(ax, no_skill, f"No-skill = {no_skill:.3f}", fontsize=11)
    ax.annotate("±SD, 8 column orderings\n(not a confidence interval)",
                xy=(1.0, 1.0), xycoords="axes fraction", xytext=(-4, -3), textcoords="offset points",
                va="top", ha="right", fontsize=10.5, color=INK_MUTED, style="italic", linespacing=1.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    _style_axes(ax, "Held-out PR-AUC (mean ± SD)", (0, 0.78))

    # No title/subtitle at top -- moved below the chart as a short explanation line instead.
    fig.text(0.055, 0.105, "Scored subset, n=2,753 (test: 592 comments)",
             ha="left", va="bottom", fontsize=11, color=INK_SECONDARY, linespacing=1.3)
    _footer(fig, "Source: 06_strategy_ablation.ipynb, cell 17 (headline_df). Column-order sensitivity documented in cells 11–14.")

    fig.subplots_adjust(left=0.29, right=0.90, top=0.95, bottom=0.22)
    _save(fig, "figure2_strategy_incremental_value")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- Rhetorical-strategy gain under length matching
# Source: this work/06_strategy_ablation.ipynb, cells 32 and 35
# ---------------------------------------------------------------------------
def figure3():
    labels = ["Full scored population\n(n = 2,753)", "Length-matched sample\n(n = 1,235)"]
    gains = [0.0664, 0.0557]
    colors = [COLOR_BLUE, COLOR_ORANGE]
    retention_pct = 0.0557 / 0.0664 * 100  # 83.9%, reported as "84%"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, gains, color=colors, height=0.40, zorder=3)

    for bar, val in zip(bars, gains):
        ax.text(val + 0.0015, bar.get_y() + bar.get_height() / 2, f"+{val:.4f} PR-AUC",
                va="center", ha="left", fontsize=14, color=INK_PRIMARY, fontweight="bold")

    bracket_x = 0.080
    ax.plot([bracket_x, bracket_x], [0, 1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.004, bracket_x], [0, 0], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.plot([bracket_x - 0.004, bracket_x], [1, 1], color=INK_MUTED, linewidth=1.3, zorder=2, clip_on=False)
    ax.text(bracket_x + 0.004, 0.5, f"{retention_pct:.0f}% retained",
            va="bottom", ha="left", fontsize=14, color=INK_PRIMARY, fontweight="bold", clip_on=False)
    ax.text(bracket_x + 0.004, 0.5, "descriptive estimate, not a significance test",
            va="top", ha="left", fontsize=8.5, color=INK_MUTED, style="italic", clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    _style_axes(ax, "Incremental PR-AUC", (0, 0.105))

    _footer(fig, "Matched by word-count decile bin, not exact-length pairs. A separate, non-comparable sample exists\n"
                 "in 04.3 (not used here). Source: 06_strategy_ablation.ipynb, cells 32, 35.")

    fig.subplots_adjust(left=0.27, right=0.92, top=0.95, bottom=0.17)
    _save(fig, "figure3_strategy_gain_length_matching")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- Comment length by outcome
# Source: this work/04.2_baseline_comparison.ipynb, cells 30-31
# ---------------------------------------------------------------------------
def figure4():
    df = _load_comment_word_counts()
    mean_non_convincing = 78.0   # full dataset, cell 30
    mean_convincing = 216.7      # full dataset, cell 30
    n_non_convincing = int((df["is_convincing"] == 0).sum())
    n_convincing = int((df["is_convincing"] == 1).sum())
    n_total = len(df)

    trimmed = df[df["word_count"] < 600]  # same < 600 cutoff as cell 31 (excludes 56 comments, 1.4%)
    bins = np.arange(0, 650, 50)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(trimmed.loc[trimmed["is_convincing"] == 0, "word_count"], bins=bins, density=True,
            color=COLOR_BLUE, alpha=0.55, zorder=3,
            label=f"No Δ (n={n_non_convincing:,}, {n_non_convincing/n_total*100:.1f}%)")
    ax.hist(trimmed.loc[trimmed["is_convincing"] == 1, "word_count"], bins=bins, density=True,
            color=COLOR_ORANGE, alpha=0.55, zorder=4,
            label=f"Δ Awarded (n={n_convincing:,}, {n_convincing/n_total*100:.1f}%)")

    ax.axvline(mean_non_convincing, color=COLOR_BLUE, linestyle=(0, (5, 3)), linewidth=1.6, zorder=5)
    ax.axvline(mean_convincing, color=COLOR_ORANGE, linestyle=(0, (5, 3)), linewidth=1.6, zorder=5)
    ax.annotate(f"mean {mean_non_convincing:.0f}", xy=(mean_non_convincing, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-6, -4), textcoords="offset points", va="top", ha="right",
                fontsize=12, color=COLOR_BLUE, fontweight="bold")
    ax.annotate(f"mean {mean_convincing:.0f}", xy=(mean_convincing, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -4), textcoords="offset points", va="top", ha="left",
                fontsize=12, color=COLOR_ORANGE, fontweight="bold")

    ax.legend(loc="upper right", frameon=False, fontsize=13, bbox_to_anchor=(1.0, 0.92))
    ax.set_xlim(0, 600)
    ax.set_ylabel("Density", fontsize=13, color=INK_SECONDARY, labelpad=8)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.set_xlabel("Comment word count (trimmed at 600 words)", fontsize=13, color=INK_SECONDARY, labelpad=8)
    ax.tick_params(axis="x", length=0, labelsize=12)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    _header(fig, "Convincing Comments Are Much Longer",
            "Word-count distribution by outcome, full dataset (n=3,971)",
            show_title=False)
    fig.text(0.07, 0.084, "Density normalized per class · Display truncated at 600 words (1.4% excluded) · Means from full dataset",
             ha="left", va="bottom", fontsize=11, color=INK_SECONDARY, linespacing=1.3)
    _footer(fig, "Source: 04.2_baseline_comparison.ipynb, cells 30–31.")

    fig.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.22)
    _save(fig, "figure4_comment_length_by_outcome")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 -- Refutation vs. justification under length control
# Source: this work/07_strategy_interpretation.ipynb, cell 36
# ---------------------------------------------------------------------------
def figure5():
    # (strategy, mean |SHAP| full population, mean |SHAP| length-matched), word_count excluded
    # (1.0245 / 0.8402 -- dwarfs every strategy and is already established in Figures 1-4).
    data = [
        ("refutation", 0.6034, 0.5193),
        ("concession", 0.3439, 0.3445),
        ("simplification", 0.3193, 0.3383),
        ("evidence", 0.3592, 0.3090),
        ("call_to_action", 0.1837, 0.2707),
        ("personal_narrative", 0.1853, 0.2168),
        ("justification", 0.4156, 0.1956),
        ("question_asking", 0.1518, 0.1806),
    ]
    # Sort by length-matched value descending, same convention as the notebook's own table.
    data = sorted(data, key=lambda row: row[2], reverse=True)
    highlighted = {"refutation", "justification"}
    display_names = {
        "refutation": "Refutation", "justification": "Justification", "concession": "Concession",
        "simplification": "Simplification", "evidence": "Evidence", "call_to_action": "Call to Action",
        "personal_narrative": "Personal Narrative", "question_asking": "Question Asking",
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = list(range(len(data)))[::-1]

    # Concession's two values (0.3439 / 0.3445) are only 0.0006 apart -- close enough that the
    # two markers would render as a single indistinguishable blob at this marker size. A small,
    # symmetric vertical nudge (not a change to either plotted value) keeps both points visibly
    # distinct rather than implying a missing observation; documented in FIGURE_NOTES.md.
    row_nudge = {"concession": 0.10}

    for y, (name, full, matched) in zip(y_pos, data):
        is_hi = name in highlighted
        dy = row_nudge.get(name, 0.0)
        line_color = INK_MUTED if not is_hi else INK_SECONDARY
        ax.plot([full, matched], [y - dy, y + dy], color=line_color, linewidth=(2.2 if is_hi else 1.2), zorder=2, alpha=(1.0 if is_hi else 0.6))
        ax.scatter([full], [y - dy], s=(95 if is_hi else 55), color=COLOR_BLUE, zorder=3,
                   edgecolors=SURFACE, linewidths=1, alpha=(1.0 if is_hi else 0.75))
        ax.scatter([matched], [y + dy], s=(95 if is_hi else 55), color=COLOR_ORANGE, zorder=3,
                   edgecolors=SURFACE, linewidths=1, alpha=(1.0 if is_hi else 0.75))

    ax.set_yticks(y_pos)
    labels = [display_names[name] for name, _, _ in data]
    ax.set_yticklabels(labels, fontsize=12.5)
    for tick, (name, _, _) in zip(ax.get_yticklabels(), data):
        if name in highlighted:
            tick.set_fontweight("bold")
            tick.set_color(INK_PRIMARY)
        else:
            tick.set_color(INK_MUTED)

    # Value callouts for the two highlighted strategies only, to keep the chart simplified.
    for name, full, matched in data:
        if name in highlighted:
            y = dict(zip([n for n, _, _ in data], y_pos))[name]
            ax.annotate(f"{full:.3f}", xy=(full, y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9.5, color=COLOR_BLUE, fontweight="bold")
            ax.annotate(f"{matched:.3f}", xy=(matched, y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9.5, color=COLOR_ORANGE, fontweight="bold")

    legend_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", color=COLOR_BLUE, markersize=9, label="Full population"),
        mlines.Line2D([], [], marker="o", linestyle="None", color=COLOR_ORANGE, markersize=9, label="Length-matched"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=10.5)

    _style_axes(ax, "Mean |SHAP value| (word_count excluded)", (0, 0.72))

    _footer(fig, "Fitted-model importance, not a causal effect. Source: 07_strategy_interpretation.ipynb, cell 36.")

    fig.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.17)
    _save(fig, "figure5_refutation_vs_justification_shap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 -- Strategy-ranking differences between datasets (CMV vs. WA)
# Sources: 07_strategy_interpretation.ipynb cell 39; 08_wa_external_validation.ipynb cell 33
# ---------------------------------------------------------------------------
def figure6():
    # (strategy, home r, home significant, wa r, wa significant)
    data = [
        ("justification", 0.3584, True, 0.1962, True),
        ("refutation", 0.3149, True, 0.0949, True),
        ("simplification", 0.1944, True, 0.1105, True),
        ("call_to_action", 0.1584, True, 0.0305, True),
        ("concession", 0.1481, True, 0.0542, True),
        ("evidence", 0.1018, True, 0.1467, True),
        ("question_asking", 0.0069, False, -0.0135, False),
        ("personal_narrative", -0.0217, False, 0.0224, True),
    ]
    display_names = {
        "refutation": "Refutation", "justification": "Justification", "concession": "Concession",
        "simplification": "Simplification", "evidence": "Evidence", "call_to_action": "Call to Action",
        "personal_narrative": "Personal Narrative", "question_asking": "Question Asking",
    }
    home_rank = {name: i + 1 for i, (name, r, *_ ) in enumerate(sorted(data, key=lambda row: row[1], reverse=True))}
    wa_rank = {name: i + 1 for i, (name, _, _, r, _) in enumerate(sorted(data, key=lambda row: row[3], reverse=True))}

    highlighted = {"refutation": COLOR_ORANGE, "evidence": COLOR_AQUA, "justification": COLOR_BLUE}
    # refutation: drops from #2 (home) to #4 (WA) despite leading by SHAP at home (Figure 5).
    # evidence: the largest riser, #6 (home) to #2 (WA).
    # justification: ranks #1 in both datasets by this metric -- the stable leader, highlighted
    # so it isn't read as "the strategy nobody is pointing at."

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x_home, x_wa = 0.0, 1.0

    for name, home_r, home_sig, wa_r, wa_sig in data:
        color = highlighted.get(name, COLOR_GRAY_LINE)
        is_hi = name in highlighted
        ax.plot([x_home, x_wa], [home_rank[name], wa_rank[name]],
                color=color, linewidth=(2.6 if is_hi else 1.4), zorder=(4 if is_hi else 2),
                alpha=(1.0 if is_hi else 0.85))
        for x, r, sig in [(x_home, home_r, home_sig), (x_wa, wa_r, wa_sig)]:
            rank = home_rank[name] if x == x_home else wa_rank[name]
            ax.scatter([x], [rank], s=(110 if is_hi else 70),
                       facecolors=(color if sig else SURFACE), edgecolors=color, linewidths=1.6,
                       zorder=(5 if is_hi else 3))
        label_color = INK_PRIMARY if is_hi else INK_MUTED
        weight = "bold" if is_hi else "normal"
        ax.annotate(display_names[name], xy=(x_home, home_rank[name]), xytext=(-10, 0),
                    textcoords="offset points", va="center", ha="right", fontsize=11,
                    color=label_color, fontweight=weight)
        ax.annotate(f"{display_names[name]}  (r={wa_r:.2f})" if is_hi else f"r={wa_r:.2f}",
                    xy=(x_wa, wa_rank[name]), xytext=(10, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=(11 if is_hi else 9.5),
                    color=label_color, fontweight=weight)
        if is_hi:
            ax.annotate(f"r={home_r:.2f}", xy=(x_home, home_rank[name]), xytext=(-10, -14),
                        textcoords="offset points", va="center", ha="right", fontsize=9.5, color=label_color)

    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(8.7, 0.3)
    ax.set_xticks([x_home, x_wa])
    ax.set_xticklabels(["CMV (home)", "Winning Arguments"], fontsize=13, fontweight="bold", color=INK_PRIMARY)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", color=INK_MUTED, markersize=8,
                      markerfacecolor=INK_MUTED, label="Significant (FDR < .05)"),
        mlines.Line2D([], [], marker="o", linestyle="None", color=INK_MUTED, markersize=8,
                      markerfacecolor=SURFACE, label="Not significant"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=2, frameon=False, fontsize=9.5)

    _header(fig, "Strategy Importance Is Dataset-Dependent",
            "Rank-biserial effect size (same method, both datasets) — rank 1 = strongest",
            show_title=False, subtitle_center=True)
    _footer(fig, "Source: 07_strategy_interpretation.ipynb cell 39 (home), 08_wa_external_validation.ipynb cell 33 (WA).",
            center=True)

    fig.subplots_adjust(left=0.14, right=0.90, top=0.90, bottom=0.15)
    _save(fig, "figure6_strategy_ranking_cmv_vs_wa")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 -- Winning Arguments validation
# Source: this work/08_wa_external_validation.ipynb, cells 20, 36, 38-39
# ---------------------------------------------------------------------------
def figure7():
    labels = ["Word count\nonly", "Strategy scores\nonly", "Robust core\n(4 features)"]
    colors = [COLOR_BLUE, COLOR_AQUA, COLOR_ORANGE]
    pr_auc = [0.5767, 0.5967, 0.5919]
    pairwise = [55.59, 57.27, 58.68]
    no_skill_pr_auc = 0.5291
    chance_pairwise = 50.0
    lit_independent = (53.78, 56.66)
    lit_full = (60.59, 64.53)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    y_pos = range(len(labels))

    bars1 = ax1.barh(y_pos, pr_auc, color=colors, height=0.55, zorder=3)
    for bar, val in zip(bars1, pr_auc):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                  va="center", ha="left", fontsize=12, color=INK_PRIMARY, fontweight="bold")
    _no_skill_line(ax1, no_skill_pr_auc, f"No-skill = {no_skill_pr_auc:.3f}", fontsize=12)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=11.5)
    ax1.invert_yaxis()
    _style_axes(ax1, "Held-out PR-AUC", (0, 0.72))
    ax1.set_title("PR-AUC", fontsize=14, color=INK_SECONDARY, pad=10, loc="left")

    bars2 = ax2.barh(y_pos, pairwise, color=colors, height=0.55, zorder=3)
    for bar, val in zip(bars2, pairwise):
        ax2.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                  va="center", ha="left", fontsize=12, color=INK_PRIMARY, fontweight="bold")
    # Chance line uses the same top-anchored convention as the PR-AUC panel's no-skill line
    # (avoids sitting on top of the bottom bar, which a bottom-anchored label did previously).
    _no_skill_line(ax2, chance_pairwise, f"Chance = {chance_pairwise:.0f}%", fontsize=12)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    _style_axes(ax2, "Pairwise accuracy", (45, 68))
    ax2.set_title("Pairwise accuracy", fontsize=14, color=INK_SECONDARY, pad=10, loc="left")
    # Literature reference moved out of the plotting region entirely (no shaded bands, no
    # floating in-panel labels, which collided with the bars/chance line) -- kept instead as
    # one line in the footer below, where there's established headroom.

    _footer(fig, "Source: 08_wa_external_validation.ipynb, cells 20, 38–39.")

    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.16, wspace=0.35)
    _save(fig, "figure7_wa_external_validation")
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    print("\nAll seven figures generated.")
