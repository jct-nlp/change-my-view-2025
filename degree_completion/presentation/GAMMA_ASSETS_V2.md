# Gamma Asset Manifest V2

Matches `GAMMA_CONTENT_V2.md`'s slide numbering. All seven figures are **approved and final**
(`degree_completion/figures/`, reviewed and approved 2026-10-16) — do not regenerate, crop,
re-letter, or edit them. Each has a PNG (300 dpi, default for Gamma upload) and an SVG (use if
Gamma's image block supports vector upload for crisper projector scaling).

**Why this file exists:** the V1 Gamma export named all seven figures in `GAMMA_ASSETS.md` but
the actual export did not embed them — every main slide that should carry an approved figure must
have it uploaded as an image block before the deck is considered built. This manifest is the
checklist for that.

## Figure-to-slide mapping

| Slide | Figure file | Format | Width in layout | Gamma placement |
|---|---|---|---|---|
| 4 — The Length Confound | `degree_completion/figures/figure4_comment_length_by_outcome.png` | PNG 300dpi (SVG available) | ~65–70% | Single image block, centered, dominant element on the slide. The image carries its own title/subtitle/footer — add no slide text beyond the headline. |
| 5 — The Pivot | `degree_completion/figures/figure1_initial_model_comparison.png` | PNG 300dpi (SVG available) | ~60% | Centered image block. The one on-slide line ("word count alone beat the full engineered pipeline") goes below the image as plain text, not as an image caption. |
| 8 — Strategies Add Real Predictive Signal | `degree_completion/figures/figure2_strategy_incremental_value.png` | PNG 300dpi (SVG available) | ~65% | Large centered image. The +0.0735 bracket is already drawn on the chart — do not add a duplicate callout on the slide. |
| 9 — Does the Gain Survive Length Control? | `degree_completion/figures/figure3_strategy_gain_length_matching.png` | PNG 300dpi (SVG available) | ~60% | Centered image. The one-line on-slide caution ("still associated with length — not proof of full independence") sits below the image, small font. |
| 10 — Refutation Holds Up; Justification Fades | `degree_completion/figures/figure5_refutation_vs_justification_shap.png` | PNG 300dpi (SVG available) | ~70% | Largest image in the deck — this is the most information-dense figure; give it more room than any other slide's visual. |
| 11 — Same Method, New Corpus | `degree_completion/figures/figure6_strategy_ranking_cmv_vs_wa.png` | PNG 300dpi (SVG available) | ~60% | Centered image. The protocol tag line ("Refit on WA's own data — not zero-shot") goes above the image in a small highlighted box — this exact wording is the thing most likely to get garbled under time pressure, so place it where the presenter reads it early and deliberately. |
| 12 — Partial Replication, Honestly Bounded | `degree_completion/figures/figure7_wa_external_validation.png` | PNG 300dpi (SVG available) | ~70% | Two-panel image, centered — wider than the other figure slides since it has two panels side by side. |

## Slides needing a built, not uploaded, visual

No approved figure exists for these — build directly in Gamma using its native shape/icon tools,
or let Gamma's AI layout generate a simple schematic from the slide text. Do not source an image
file for any of these; do not invent a "figure" and present it as if it came from a notebook.

- **Slide 1 (Title & Delta Mechanism):** two-box schematic — "OP post" → arrow → "Reply (Δ
  awarded)" with a small delta badge icon. Abstract/iconographic only; no fabricated post text.
- **Slide 2 (Three Research Questions):** a simple vertical funnel/arrow diagram, three stacked
  boxes narrowing in scope, each labeled with its short RQ phrase from the slide text.
- **Slide 3 (Dataset & Task Definition):** horizontal six-step pipeline diagram — data collection
  → annotation → feature engineering → modeling → strategy scoring → external validation.
- **Slide 7 (Eight Rhetorical Strategies):** a plain horizontal row of 8 equal-weight labels, no
  chart, no highlighting of any subset — keep all eight visually equal here.
- **Slide 13 (Three Answers, Decreasing Confidence):** a simple 3-row qualitative recap with a
  status icon per row (e.g., checkmark / partial-check / arrow) — restates conclusions already
  stated earlier in the talk; do not attach any new number or invented confidence percentage to
  it.

## Slides with no visual at all

Slide 6 (Transition) is deliberately minimal — a single optional icon at most, no chart. It
functions as a spoken pause between the talk's two halves; resist the urge to fill it with a
diagram.

## Backup slides

- **B1, B2** — text/table only, no image.
- **B3** — may optionally reuse `figure7_wa_external_validation.png` as a visual backdrop while
  discussing the best-found WA configuration; this is the same file already used on Slide 12, not
  a new asset. Do not alter or re-annotate the image file itself.

## Upload checklist

1. Upload all seven PNGs (or SVGs) to their respective slides per the table above — confirm each
   slide's figure matches exactly (Figures 5 and 6 are both strategy-ranking comparisons and are
   easy to mis-pair).
2. Build the five fresh schematics/diagrams (Slides 1, 2, 3, 7, 13) directly in Gamma.
3. Confirm no slide shows an AI-auto-generated chart standing in for an approved figure — check
   Slides 4, 5, 8, 9, 10, 11, 12 specifically after any Gamma "regenerate slide" action, since
   that feature can silently replace an uploaded image with a fabricated one.
4. Do not upload, link, or reference any image or table from `wa_quick_experiments.ipynb`.
5. After upload, preview at projector size and confirm Figure 5's and Figure 7's two-panel/dense
   layouts are still legible at their enlarged on-slide widths (70%, 70%).
