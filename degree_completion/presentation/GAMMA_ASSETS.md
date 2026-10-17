# Gamma Asset Manifest

All seven figures below are **approved and final** (`degree_completion/figures/`, reviewed and
approved 2026-10-16). Do not regenerate, crop, re-letter, or edit them — upload as-is. Each has
both a PNG (300 dpi, recommended default for Gamma upload) and an SVG (use instead if Gamma's
image block supports vector upload and you want crisper scaling on a projector).

No other image assets exist for this deck. Three slides (1, 3, 7) call for a fresh schematic that
does not exist anywhere in the repository — build those directly in Gamma (see "Slides needing a
built, not uploaded, visual" below); do not search for or invent a source file for them.

## Figure-to-slide mapping

| Slide | Figure file | Format | Gamma placement |
|---|---|---|---|
| 4 — The Length Confound | `degree_completion/figures/figure4_comment_length_by_outcome.png` | PNG 300dpi (SVG available) | Single image block, 60–65% slide width, centered. Image already carries its own title/subtitle/footer — do not overlay additional text on it. Bullets go beside or below, not on top. |
| 5 — The Pivot: Length Beats the Engineered Pipeline | `degree_completion/figures/figure1_initial_model_comparison.png` | PNG 300dpi (SVG available) | Image block, ~55% slide width, left or center. Keep the bolded framing line ("this wasn't a failure…") as slide text outside the image, not as a caption inside it. |
| 8 — Strategy Scores Add Real Predictive Signal | `degree_completion/figures/figure2_strategy_incremental_value.png` | PNG 300dpi (SVG available) | Large centered image, ~65% width. Methodological disclosure bullet goes below in smaller text. |
| 9 — Does the Gain Survive Length Control? | `degree_completion/figures/figure3_strategy_gain_length_matching.png` | PNG 300dpi (SVG available) | Centered image, ~60% width. Keep the `04.3` aside as a verbal-only note (speaker notes), not on the slide or image. |
| 10 — Refutation Holds Up; Justification Doesn't | `degree_completion/figures/figure5_refutation_vs_justification_shap.png` | PNG 300dpi (SVG available) | Large central image, ~65% width — this is the most information-dense figure; give it the most space of any slide in the deck. Reliability caveat bullet set apart/highlighted below. |
| 11 — External Validation: Protocol and the Reversal | `degree_completion/figures/figure6_strategy_ranking_cmv_vs_wa.png` | PNG 300dpi (SVG available) | Centered image, ~60% width. Put the protocol statement in a highlighted text box **above** the image, not below — its exact wording is the thing most likely to get garbled under time pressure, so it should be read early and deliberately. |
| 12 — External Validation: Quantitative Results | `degree_completion/figures/figure7_wa_external_validation.png` | PNG 300dpi (SVG available) | Two-panel image (PR-AUC left, pairwise accuracy right), centered, ~70% width — it's wider than the others. Limitation line in italics below. |

## Slides needing a built, not uploaded, visual

These three slides call for a simple diagram per `PRESENTATION_STORYLINE.md` / `FIGURE_SELECTION_
PLAN.md`, but no such file exists anywhere in the repository (confirmed in `FIGURE_SELECTION_
PLAN.md` Sections 1–2: both explicitly state "No" under "Exists already?"). Build these directly
in Gamma using its native shape/icon/diagram tools, or let Gamma's AI layout generate a simple
schematic from the slide text — do not source an image file for these.

- **Slide 1 (Title & Delta Mechanism):** a simple two-box schematic — "OP post" → arrow →
  "Reply (Δ awarded)" with a small delta badge icon on the second box. Keep it abstract/
  iconographic. Do not fabricate real post or comment text.
- **Slide 3 (Dataset & Task Definition):** a horizontal six-step pipeline diagram — data
  collection → annotation → feature engineering → modeling → strategy scoring → external
  validation — six boxes with arrows, each labeled with its source notebook in small text.
- **Slide 7 (Eight Rhetorical Strategies):** optional only — a plain horizontal row of 8 labels
  (justification, concession, evidence, refutation, personal_narrative, call_to_action,
  simplification, question_asking), equal visual weight, no chart. A text bullet list works
  equally well if a visual anchor isn't needed; per `FIGURE_SELECTION_PLAN.md` this is marked
  "optional," not essential.

## Slides with no visual at all

Slides 2 (Research Questions), 6 (Transition), and 13 (Conclusions) are intentionally text-only
per the storyline — slide 6 especially should stay minimal (large centered text, generous
whitespace) since it functions as a spoken pause between the two halves of the talk.

## Backup slides

- **B1, B2** — text/table only, no image.
- **B3** — may optionally reuse `figure7_wa_external_validation.png` as a visual backdrop/
  reference while discussing the best-found WA configuration; this is the same file already used
  on Slide 12, not a new asset. Do not alter or re-annotate the image file itself.

## Upload checklist before the Oct 26 "complete deck" milestone

1. Upload all seven PNGs (or SVGs) listed above to their respective slides — confirm each slide's
   figure matches this table exactly before moving on (easy to mis-pair Figures 5 and 6, since
   both are strategy-ranking comparisons).
2. Build the two required schematics (Slides 1 and 3) and the optional strategy-label row
   (Slide 7) directly in Gamma.
3. Do not upload, link, or reference any image or table from `wa_quick_experiments.ipynb` —
   that notebook's entire output is out of scope for this deck, including backup slides.
4. After upload, sanity-check on a projector-sized preview (not just a laptop screen) that
   Figure 5's and Figure 7's two-panel layouts are still legible — these are the densest images
   in the set.
