# Figure Selection Plan — MSc Defense Presentation

**Scope:** planning and evidence review only. No figures generated or exported, no slides built, no
notebooks or code modified, no experiments rerun, no research conclusions changed. Built by reading
`PRESENTATION_STORYLINE.md`, `RESULTS_SHORTLIST.md`, and the cell-level content of the relevant
notebooks in `this work/` directly (cell numbers below are 0-indexed, matching the `.ipynb` JSON
structure).

**How to read each entry:** one visual candidate per numbered item, grouped by the ten presentation
sections. "Exists" means a plot or table is already produced somewhere in a notebook's output;
every such plot exists only as an *embedded* image inside the `.ipynb` file (no standalone PNG/SVG
exists anywhere in the repo) and every such table exists only as a live `DataFrame` output, so even
an "export as-is" item still requires a short extraction/re-render step before it can go on a slide.

---

## Section 1 — Problem, motivation, research questions (RQ: all)

**Main message:** explain CMV and the delta mechanism, then state the three research questions.

| | |
|---|---|
| **Recommended visual** | A simple annotated mockup: an OP post + a reply with a delta badge. |
| **Exists already?** | **No.** `00_introduction.ipynb` has only 2 cells, both prose (§1.1–§1.4), no image or diagram cell anywhere in the notebook. |
| **Source** | None — conceptual, not data-derived. |
| **Demonstrates** | The forum mechanic the whole project depends on: a delta is a community-awarded, per-comment success label. |
| **Key numbers it must show** | None — purely illustrative; do not put fabricated post/comment text on it. |
| **Export vs. redraw** | New, build from scratch. |
| **Effort** | Low (~30–45 min in slide software; simplest option is a one-off schematic, not a screenshot of a real post). |
| **Priority** | **Useful**, not essential — a text/bullet slide works as a fallback if time is tight. |
| **Caveats** | None scientific. Keep it purely explanatory; don't let it imply the delta measures "true" persuasion (that nuance belongs in Section 10's limitations, not this slide). |

---

## Section 2 — Dataset construction and task definition (RQ1)

| | |
|---|---|
| **Recommended visual** | A small pipeline diagram: data collection → annotation → feature engineering → modeling → strategy scoring → external validation. |
| **Exists already?** | **No.** `00_introduction.ipynb` §1.4 describes the 5 stages in prose only; `REPORT_OUTLINE.md` §2.5 independently flags this exact diagram as `[MISSING]`. |
| **Source** | `00_introduction.ipynb` §1.4 (prose description to adapt); `01_data_download.ipynb` and `02_data_annotation.ipynb` (actual stage boundaries). |
| **Demonstrates** | The end-to-end structure of the project, so later sections' notebook references make sense. |
| **Key numbers it must show** | None — a process diagram, not a results chart. If a dataset-size callout is added to the same slide: 3,971 comments, 641 delta-awarded (16.1%). |
| **Export vs. redraw** | New, build from scratch. |
| **Effort** | Low–medium (~1 hr; six boxes and arrows, label each with its notebook). |
| **Priority** | **Useful** — supports orientation for RQ1 but isn't itself part of the scientific argument. |
| **Caveats** | The exact 3,971 / 641 / 16.1% figures are not printed together in a single notebook cell — they were independently computed by reading `results/02_comments_df_base.csv.zip` directly (see Verification Notes below). `02_data_annotation.ipynb` cell 37 prints the final row count (3,971) after all three cleanup drops; the 641/16.1% split must be computed from the saved CSV, not assumed. |

---

## Section 3 — Initial experiments and the length finding (RQ1)

### 3a. Word-count-by-outcome histogram — **essential**

| | |
|---|---|
| **Recommended visual** | Histogram of `word_count`, split/colored by `is_convincing`, comments <600 words. |
| **Exists already?** | **Yes**, as embedded notebook output. |
| **Source notebook/cell** | `04.2_baseline_comparison.ipynb`, cell 31 (`plot_feature(word_count_trimmed, 'word_count', ...)`), built on the computation in cell 30. |
| **Demonstrates** | Convincing comments are systematically, substantially longer — the visual root of the length confound. |
| **Key numbers it must show** | Mean word count 78.0 (non-convincing) vs. 216.7 (convincing); r=0.3716 (p=2.92e-130) belongs in a caption or adjacent text, not necessarily on the chart itself. |
| **Export vs. redraw** | Export as-is — extract the embedded image from the notebook, or re-run `plot_feature(...)` against the saved results to get a clean, higher-resolution render (the function only calls `plt.show()`, never `savefig`, so a short extraction/re-run step is needed either way; see Verification Notes). |
| **Effort** | Low (~20–30 min to extract/re-render and crop for a slide). |
| **Priority** | **Essential.** |
| **Caveats** | Chart already excludes the 56 comments (1.4%) at ≥600 words by design (noted directly under the chart in the notebook) — keep that exclusion note nearby if a sharp question comes up. |

### 3b. Headline PR-AUC comparison table (engineered vs. TF-IDF vs. word-count-alone) — **essential**

| | |
|---|---|
| **Recommended visual** | Simple bar chart or table: Majority-class, Stratified-random, Selected engineered pipeline, TF-IDF+LogReg, `word_count` alone — PR-AUC (+ no-skill baseline line at 0.161/0.203 test-fold rate). |
| **Exists already?** | Partially — as two separate live tables, not yet combined into one chart. |
| **Source notebook/cell** | `04.2_baseline_comparison.ipynb` cell 9 (`baseline_results` — majority/stratified/selected pipeline) + cell 10 (adds TF-IDF+LogReg) + cell 33 (`length_vs_others` — adds `word_count` alone). |
| **Demonstrates** | The central pivot: a bare-minimum baseline beat the engineered pipeline, and length alone beat everything. |
| **Key numbers it must show** | Engineered 0.3848, TF-IDF+LogReg 0.5119, `word_count` alone 0.5256, Majority-class 0.2031, Stratified-random 0.2013. |
| **Export vs. redraw** | Redraw as a simple horizontal bar chart — the three live tables are correct but scattered across cells and mix metrics (PR-AUC/ROC-AUC/F1); a slide needs just PR-AUC, pulled into one clean comparison. |
| **Effort** | Low (~30 min — five numbers, one bar chart). |
| **Priority** | **Essential.** |
| **Caveats** | `04.2` cell 12–14's column-order/bootstrap-CI check shows the Selected-pipeline point estimate (0.3848) sits about 1 SD above its own order-robust mean (0.3676±0.0182); most pairwise CIs here overlap substantially — only the word-count-vs-engineered comparison (Section 3c below) clears that bar decisively. Say "point estimate" if asked, not "the only true value." |

### 3c. Word-count-fix comparison (0.3848 → 0.4922) — **useful, folds into 3b or stands alone**

| | |
|---|---|
| **Recommended visual** | Before/after bar pair: engineered pipeline without vs. with `word_count` folded in. |
| **Exists already?** | Yes, as a live table. |
| **Source notebook/cell** | `04.2_baseline_comparison.ipynb` cell 77 (`selection_comparison`) and cell 79 (`selection_robustness_df`, adds bootstrap CIs + order-robust mean±SD). |
| **Demonstrates** | The single most statistically decisive comparison in the project — CIs barely overlap. |
| **Key numbers it must show** | PR-AUC 0.3848 → 0.4922 (order-robust: 0.3676±0.0182 → 0.4831±0.0257); bootstrap 95% CIs [0.311, 0.463] vs. [0.410, 0.586]. |
| **Export vs. redraw** | Redraw as a simple two-bar comparison with CI whiskers. |
| **Effort** | Low (~20 min). |
| **Priority** | **Useful** — strengthens Section 3's close but is a secondary point next to 3a/3b; cut first if time is short. |
| **Caveats** | F1 numbers in the same table are *not* comparable (default 0.5 threshold here vs. a validation-tuned threshold elsewhere) — omit F1 from the slide to avoid an easy, wrong "gotcha" comparison. |

---

## Section 4 — Central question: beyond length? (RQ2)

| | |
|---|---|
| **Recommended visual** | **None.** |
| **Exists already?** | N/A. |
| **Rationale** | This is an explicit 30-second transition in the timing table — a text-only statement of the next question reads better here than manufacturing a chart for a pivot sentence. |
| **Priority** | N/A (no visual recommended). |

---

## Section 5 — Eight strategies and measurement (RQ2–3)

| | |
|---|---|
| **Recommended visual** | None essential; at most a compact 8-label row (icons or a simple list) as a visual anchor. |
| **Exists already?** | Strategy *distributions* exist as 7 separate histograms (one per strategy, `plot_category_histogram`), but these are diagnostic, not a compact overview, and showing all 8 would violate the "keep this section brief" guidance. |
| **Source notebook/cell** | `07_strategy_interpretation.ipynb` cells 6–13 (one `plot_category_histogram` call per strategy, skipping `justification` which is plotted earlier in the same cell run). Rubric definitions: `05_rhetorical_scoring.ipynb` cell 1. |
| **Demonstrates** | N/A for the main slide — these are not the recommended visual, listed here only because they're the closest existing candidate and were explicitly rejected. |
| **Export vs. redraw** | N/A. |
| **Effort** | N/A. |
| **Priority** | **Optional** — a plain 8-item label list (no chart) is enough; do not use the 7 individual histograms here, they belong to a different, more detailed level of discussion than this section's allotted 1:10. |
| **Caveats** | If any reliability number is shown here (test-retest 95–99%), source it to `05_rhetorical_scoring.ipynb` cell 31 (summary) / cell 30 (table) — and note it's a lightweight check (N=200 retest, N=18 hand-scored), not a formal reliability study. |

---

## Section 6 — Strategy-combination results (RQ2)

| | |
|---|---|
| **Recommended visual** | Bar/dot chart: current model alone vs. +8 strategy scores vs. strategy-scores-alone, each with its mean±SD error bar, plus the no-skill baseline line. |
| **Exists already?** | As a live table, not a chart. |
| **Source notebook/cell** | `06_strategy_ablation.ipynb` cell 17 (`headline_df`) — built on the order-robust evaluator defined in cells 10–15. |
| **Demonstrates** | The project's cleanest "yes" result: adding strategy scores is a real, order-robust gain. |
| **Key numbers it must show** | 0.4467±0.0196 → 0.5202±0.0177 (+0.0735); strategy-only 0.3538±0.0079; no-skill baseline 0.2382 (this test fold's own positive rate, per `06`'s convention — not the whole-dataset 0.161). |
| **Export vs. redraw** | Redraw — the live table has no corresponding chart yet; a 3-bar chart with error bars is new work, not an export. |
| **Effort** | Low–medium (~30–45 min — needs the mean±SD values pulled correctly, see caveat). |
| **Priority** | **Essential.** |
| **Caveats** | Column-order sensitivity is why these numbers are mean±SD at all (cells 11–14: reordering 237 identical columns swings PR-AUC by up to 0.0422) — the chart's error bars **are** that sensitivity, not sampling noise, and should be captioned as such if an examiner asks what the error bars represent. |

---

## Section 7 — Length matching and compact model (RQ2)

### 7a. Strategy-gain retention under length control — **essential**

| | |
|---|---|
| **Recommended visual** | Simple 2-group bar chart: strategy-score PR-AUC gain, full scored population vs. length-matched. |
| **Exists already?** | As a live table. |
| **Source notebook/cell** | `06_strategy_ablation.ipynb` cell 35 (`length_benchmark_df`), built from the decile-matching procedure in cell 32. |
| **Demonstrates** | The strategy-score gain survives length control far better than the feature families it's added to. |
| **Key numbers it must show** | Strategy gain 0.0664 (full, n=2,753) → 0.0557 (length-matched) — **84% retention**; for contrast in the same slide's caption only (not necessarily the chart itself), `word_count`'s own lift and the full pipeline's own lift both retain far less (~18% and ~25% respectively, per `04.3`/`06`, see caveat below on which table each number comes from). |
| **Export vs. redraw** | Redraw as a 2-bar (or 2×2 grouped-bar) chart from the live table. |
| **Effort** | Low–medium (~30–45 min). |
| **Priority** | **Essential.** |
| **Caveats — important, verify before combining with other retention numbers:** this length-matched sample is built **within the 2,753-row scored population** (`06` cell 32, explicitly "within the scored population"; `07` cell 32 confirms the same construction is reused there). This is a **different sample** from `04.3_length_controlled_analysis.ipynb`'s own strict 1:1 match (1,235 rows, built from the full 3,971-row dataset, `04.3` cell 19–21). The two notebooks' "length-matched" tables are **not the same population** and must not be silently merged into a single chart or table as if they were — if both the ~18%/~25% word-count/engineered-pipeline retention figures (from `04.3`) and the 84% strategy-gain retention figure (from `06`) appear on the same slide, caption them as two separate, compatible-in-spirit-but-different-in-construction checks, not one unified experiment. |

### 7b. Nine-feature compact model vs. full feature set — **useful**

| | |
|---|---|
| **Recommended visual** | Small 2-bar or single-row table: 9-column combination vs. full 237-column set. |
| **Exists already?** | As numbers inside prose/markdown summaries, not a dedicated chart. |
| **Source notebook/cell** | `06_strategy_ablation.ipynb` cell 22–23 (cumulative family build, identifies `strategy scores + word_count` as the best combination) and cell 30 (9-column re-selection: RF 0.5432±0.0065 vs. XGB 0.5380±0.0108, order-robust). |
| **Demonstrates** | A small, interpretable 9-feature model is not just competitive with but better than the full engineered+strategy set — a parsimony argument. |
| **Key numbers it must show** | 9-feature PR-AUC 0.5379–0.5380±0.01 vs. full 237-feature 0.5202±0.0177. |
| **Export vs. redraw** | Redraw (new, simple). |
| **Effort** | Low (~20 min). |
| **Priority** | **Useful** — strong if there's time or as a backup slide for the "why not throw in everything" question; not required for the core RQ2/RQ3 arc. |
| **Caveats** | The architecture note matters: cell 30 shows XGB and RF are statistically indistinguishable on this 9-column set (gap smaller than the order-noise band) — if shown, say "XGB (0.5380), though Random Forest scores nearly identically here," not "XGB is best." |

---

## Section 8 — Refutation vs. justification (RQ3)

### 8a. Simplified before/after SHAP-importance comparison — **essential**

| | |
|---|---|
| **Recommended visual** | Simplified horizontal bar/dot chart: mean |SHAP| per strategy, full population vs. length-matched, side by side — **not** the two raw beeswarm plots. |
| **Exists already?** | The underlying numbers exist as a live comparison table; the existing plots are dense beeswarms (one feature per row, one dot per test-set comment), which will not read on a projector per the project's own guidance. |
| **Source notebook/cell** | Numbers: `07_strategy_interpretation.ipynb` cell 36 (`shap_comparison` table). Underlying (not recommended for direct reuse) beeswarm images: cell 23 (full population) and cell 35 (length-matched). |
| **Demonstrates** | `refutation` holds its SHAP rank under length control; `justification`'s collapses — the project's most specific, examinable finding. |
| **Key numbers it must show** | Full population: `word_count` 1.0245, **refutation 0.6034**, justification 0.4156, evidence 0.3592, concession 0.3439, simplification 0.3193, personal_narrative 0.1853, call_to_action 0.1837, question_asking 0.1518. Length-matched: `word_count` 0.8402, **refutation 0.5193**, concession 0.3445, simplification 0.3383, evidence 0.3090, call_to_action 0.2707, personal_narrative 0.2168, **justification 0.1956**, question_asking 0.1806. |
| **Export vs. redraw** | Redraw — a simplified paired bar chart built from the already-tabulated `shap_comparison` DataFrame, explicitly instead of the two beeswarms (this is the clearest instance in the whole project of "prefer a simplified comparison chart over a complex plot"). |
| **Effort** | Medium (~45–60 min — 9 features × 2 conditions, needs a legible ranked layout, e.g. sorted by length-matched value). |
| **Priority** | **Essential.** |
| **Caveats** | SHAP importance and the Mann-Whitney effect size (8b below) measure different things — don't caption this chart as "the" importance ranking without noting the model-free cross-check exists too. Neither is a causal-influence claim. |

### 8b. Mann-Whitney effect-size comparison — **useful, pairs with 8a**

| | |
|---|---|
| **Recommended visual** | Table (not necessarily a chart) of rank-biserial r, full population vs. length-matched, flagging FDR-significance. |
| **Exists already?** | Yes, as a live table. |
| **Source notebook/cell** | `07_strategy_interpretation.ipynb` cell 39 (`effect_size_comparison`). |
| **Demonstrates** | A model-free cross-check that broadly agrees with 8a: refutation/justification/simplification are the only strategies that stay significant after matching. |
| **Key numbers it must show** | Full: justification r=0.3584, refutation r=0.3149, simplification r=0.1944 (others smaller/n.s.). Matched: refutation r=0.154 (49% retained), justification r=0.131 (37% retained), simplification r=0.079; `personal_narrative` flips to r=−0.086. |
| **Export vs. redraw** | Export the table as-is (already clean and compact) or lightly restyle for slide typography. |
| **Effort** | Low (~15–20 min). |
| **Priority** | **Useful** — good as a compact supporting table under 8a's chart, or held as a backup-slide answer to "is this robust to a different method?" |
| **Caveats** | State directly next to this table: refutation and justification are the two strategies `05_rhetorical_scoring.ipynb` (cell 31 test-retest summary; cell 35/37 hand-scoring) flags as hardest to calibrate, including against a human judge. |

---

## Section 9 — WA external validation and surprise (RQ3)

### 9a. Cross-dataset strategy-ranking reversal — **essential**

| | |
|---|---|
| **Recommended visual** | Two small ranked lists or a paired bar chart: top strategies by effect size, home dataset vs. WA, visually showing refutation/justification swapping places. |
| **Exists already?** | As two separate live tables on two different datasets, never combined into one comparison view. |
| **Source notebook/cell** | Home: `07_strategy_interpretation.ipynb` cell 39 (`effect_size_comparison`, "full population" column). WA: `08_wa_external_validation.ipynb` cell 33 (`effect_size_comparison`, WA's own "full population" column). |
| **Demonstrates** | The headline surprise: the structural finding (strategies matter beyond length) replicates; the specific top strategy does not. |
| **Key numbers it must show** | Home: refutation top by SHAP (0.6034), justification top by effect size (r=0.3584). WA: justification top by **both** SHAP and effect size (r=0.1962); refutation 4th by effect size, near-bottom by SHAP. |
| **Export vs. redraw** | Redraw — this comparison does not exist as a single artifact anywhere; it must be built by placing the two notebooks' tables side by side. |
| **Effort** | Medium (~45 min — requires carefully aligning comparable columns from two different notebooks; see caveat). |
| **Priority** | **Essential.** |
| **Caveats** | Confirm which effect-size metric is being compared across datasets — both `effect_size_comparison` tables use the same rank-biserial method and the same FDR correction, so this specific comparison is methodologically compatible. Do **not** extend the same side-by-side treatment to raw PR-AUC numbers (see 9b) without the WA-specific baseline adjustment described there. |

### 9b. WA feature-set / pairwise-accuracy comparison vs. literature benchmarks — **essential**

| | |
|---|---|
| **Recommended visual** | Table: `word_count` only / strategy scores only / robust 4-feature core / best-found (−call_to_action) — PR-AUC and pairwise accuracy — with the two literature reference bands (Independent Scoring, full MS-PS method) shown alongside, not merged into the same bars. |
| **Exists already?** | As a live table (pairwise accuracy) plus a separate PR-AUC table; not yet combined with the literature numbers on one slide. |
| **Source notebook/cell** | `08_wa_external_validation.ipynb` cell 36 (PR-AUC pruning/configuration table) and cell 39 (`pairwise_accuracy_df`). Literature reference points are quoted in the notebook's own markdown (cell 38) from Labruna et al. (2026), Table 1. |
| **Demonstrates** | WA's own structural replication, landing in an honest, bounded position relative to the literature. |
| **Key numbers it must show** | `word_count` only: PR-AUC 0.5767, pairwise 55.59%. Strategy scores only: PR-AUC 0.5967, pairwise 57.27%. Robust 4-feature core: PR-AUC 0.5919, pairwise 58.68%. Best found (−call_to_action): pairwise 59.10%. Literature: Independent Scoring 53.78–56.66%, full MS-PS method 60.59–64.53%. |
| **Export vs. redraw** | Redraw as a clean comparison table (the two live tables are correct but use different column sets; combine only the comparable columns). |
| **Effort** | Medium (~30–45 min). |
| **Priority** | **Essential** — this is the slide more presentation-friendly than another SHAP plot at this point in the talk, per the storyline's own framing. |
| **Caveats** | **Do not draw this as, or caption it as, a frozen/zero-shot model applied to WA.** Any diagram or arrow accompanying this table must show "architecture/configuration reused, then a new model instance fit on WA's own training split, evaluated on WA's own heldout split" — not "trained model applied directly." The Labruna et al. (2026) citation and its Table 1 benchmark numbers are now **verified** against the published paper (2026-10-14) — see `REPORT_OUTLINE.md` §2.3 — so this table's literature-comparison numbers are safe to finalize. |

**Explicitly excluded from this section (and from every backup slide):** any `wa_quick_experiments.ipynb` numbers — the genuine zero-shot (no-refit) check, the embeddings-on-top check, and the re-adding-concession check — per the guardrails already in `RESEARCH_GUARDRAILS.md` and `PRESENTATION_STORYLINE.md`.

---

## Section 10 — Conclusions, contribution, limitations (RQ: all)

| | |
|---|---|
| **Recommended visual** | None required; at most a condensed 3-row recap of the RQ1–RQ3 answers (text table, reusing wording already in `PRESENTATION_STORYLINE.md`'s RQ section). |
| **Exists already?** | N/A — this is a synthesis slide, not a data visual. |
| **Source** | Synthesis of Sections 3, 6–9 above; `REPORT_OUTLINE.md` §5 for the fuller discussion-section draft. |
| **Demonstrates** | N/A. |
| **Key numbers it must show** | None new — reuses numbers already shown earlier in the talk. |
| **Export vs. redraw** | N/A. |
| **Effort** | Low (~15 min, text layout only). |
| **Priority** | **Optional** as a visual (the content itself is essential, but as a recap table, not a new figure). |
| **Caveats** | None beyond what's already stated in Sections 3/6–9. |

---

## Prioritized shortlist — 5–7 essential visuals carrying the main scientific story

1. **Word-count-by-outcome histogram** (Section 3a) — `04.2_baseline_comparison.ipynb` cell 31. Why it: the single clearest "why" visual for the length confound that opens the whole narrative (RQ1).
2. **Headline baseline-comparison chart** (Section 3b) — `04.2_baseline_comparison.ipynb` cells 9/10/33. Why it: the concrete numbers (0.3848 / 0.5119 / 0.5256) that make the length confound a hard result, not just a claim (RQ1).
3. **Strategy-combination gain chart** (Section 6) — `06_strategy_ablation.ipynb` cell 17. Why it: the project's cleanest "yes" result and the backbone of RQ2.
4. **Strategy-gain length-control retention chart** (Section 7a) — `06_strategy_ablation.ipynb` cell 35. Why it: without this, the Section 6 gain could still be "just length" — this is what answers that directly (RQ2).
5. **Refutation-vs-justification before/after SHAP comparison** (Section 8a) — built from `07_strategy_interpretation.ipynb` cell 36. Why it: the most specific, examinable finding in the project, and its own visual argument (RQ3).
6. **Cross-dataset strategy-ranking reversal** (Section 9a) — built from `07` cell 39 + `08_wa_external_validation.ipynb` cell 33. Why it: the headline surprise of external validation, and the clearest single visual for "structural replication, detail reversal" (RQ3).
7. **WA feature-set/pairwise-accuracy table vs. literature** (Section 9b) — `08_wa_external_validation.ipynb` cells 36/39. Why it: anchors WA's result to the published literature honestly, without overclaiming (RQ3).

If forced to cut to 5: drop #2 into a one-line callout inside #1's slide, and fold #7 into #6's slide as a supporting table rather than its own slide — but keep all seven as distinct line items in speaker notes even if two end up sharing a physical slide.

---

## Verification notes / open questions for the next pass

- The dataset-composition numbers (3,971 / 641 / 16.1%) are not printed together in any single
  notebook cell — `02_data_annotation.ipynb` cell 37 prints only the final row count (3,971). The
  641/16.1% split should be independently recomputed from `results/02_comments_df_base.csv.zip`
  before it appears on a slide, rather than copied from a prior document without re-checking.
- **Two different "length-matched" samples exist in this project and must not be conflated:**
  `04.3_length_controlled_analysis.ipynb`'s strict 1:1 match (1,235 rows, from the full 3,971-row
  dataset) vs. `06_strategy_ablation.ipynb`/`07_strategy_interpretation.ipynb`'s strict 1:1 match
  (built within the 2,753-row scored population). Any slide or report paragraph citing both a
  length-retention percentage for `word_count`/the engineered pipeline (from `04.3`) and the
  strategy-score gain retention (84%, from `06`) must caption them as two separate constructions,
  not one unified experiment.
- **Resolved (2026-10-14):** the Labruna et al. (2026) and Tan et al. (2016) citations used for the
  WA literature-comparison table (Section 9b) are now verified directly against the published
  papers (ACL Anthology 2026.findings-eacl.97; Tan et al., WWW 2016, pp. 613–624) — see
  `REPORT_OUTLINE.md` §2.3 for full details. The three benchmark ranges `08`'s notebook cites
  (Independent Scoring 53.78–56.66%, MS-PS-AVG 60.59–62.83%, best 64.53%) were checked against the
  paper's own Table 1 and match exactly. One minor wording nuance surfaced: the notebook describes
  Labruna et al.'s taxonomy as "6 propaganda-technique strategies on news articles," but their main
  experiments apply it to Winning Arguments/Anthropic/Persuasion for Good — the news-article
  dataset is only a secondary validation check. Not a numbers issue, just a phrasing tightening if
  this sentence is reused verbatim in the final report.
- Two existing screenshot-style images in `02_data_annotation.ipynb` (cells 16 and 31) were
  reviewed and **not** selected for any section — they are pasted debugging screenshots (a worked
  example of the comment tree, and a dataframe preview) rather than analytical charts, and don't
  suit a presentation slide as-is.
- No new numbers or figures were invented anywhere in this plan — every number above was read
  directly from the cited notebook cell, cross-checked against `RESULTS_SHORTLIST.md` where that
  document covers the same finding.
