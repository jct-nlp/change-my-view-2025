# Review Notes — Methods Chapter Draft

Notes for reviewing `02_METHODS.md`. This is a first draft for human review, not a finalized text.
All facts in the chapter were either (a) already marked `[VERIFIED]` in `REPORT_OUTLINE.md` from
the prior research-mapping pass, or (b) freshly re-verified directly against live notebook cells
during this task (via two parallel verification passes covering all twelve source notebooks). No
number in the chapter was carried forward from planning-document prose without at least one of
these two forms of grounding.

## 1. Feature-set discrepancy — RESOLVED (2026-10-20): keep as is

**`06_strategy_ablation.ipynb` does not reuse the exact feature set selected in `04.1`/`04.2`, and
the notebook's own markdown mislabels this.**

**Resolution:** at your request, I ran a standalone, read-only replication of `06`'s own pipeline
with the 11 untrusted-derived columns added back in (same data, same thread-grouped split, same
RobustScaler+SMOTE+XGBoost architecture, same order-robust column-shuffling convention), to check
whether correcting this would change the ablation's headline numbers. Result: the full model's
performance is essentially unchanged (0.5202 → 0.5169, within noise), but the engineered-only
baseline rises substantially (0.4467 → 0.4858), so the specific strategy-score gain this project
quotes everywhere as "the clearest yes result" would shrink from +0.0735 to +0.0311 — still a real,
order-robust gain, but less than half its current reported size. **You decided to keep the project
as-is**: `06_strategy_ablation.ipynb`, `RESULTS_SHORTLIST.md`, the presentation materials, and this
Methods chapter's §2.7 all continue to describe the 229/237-column baseline actually computed in
`06`, without correction, on the grounds that this level of precision exceeds what the university
report requires. No files were changed as a result of this check (the replication ran in a scratch
script, not against any notebook or committed result). This question is now closed — do not reopen
it without a concrete new reason.

The original finding, preserved below for the record:

- `04.1_model_training.ipynb`'s model-selection grid selects a 239-column "trusted+untrusted"
  feature set (41 handcrafted + 187 embeddings + 11 encoded columns from six features flagged
  "untrusted": `sentiment_vader`, `tone_label`, `style_label`, `ethos_score`, `pathos_score`,
  `logos_score`). This is the feature set underlying the headline PR-AUC 0.3848 reported in §2.3.
  `04.2_baseline_comparison.ipynb`'s word-count-augmented re-selection, in the same family, selects
  a 240-column version of the same "trusted+untrusted" set (the 239 columns plus `word_count`),
  underlying PR-AUC 0.4922 in §2.4.
- `06_strategy_ablation.ipynb`, however, builds its own "engineered features" baseline from
  scratch (cell 4), using the 228-column "trusted" set (41 handcrafted + 187 embeddings, **no**
  untrusted-derived columns) plus `word_count`, for 229 columns total. This is confirmed directly
  by the notebook's own code: no `untrusted_features`/`untrusted_encoded` variable appears
  anywhere in `06`'s code cells. Combined with the eight strategy scores, this gives the 237-column
  figure already used elsewhere in this project's documentation (`RESULTS_SHORTLIST.md` Finding 3).
- **The notebook's own markdown is internally inconsistent about this.** Two markdown cells in `06`
  (cells 6 and 8, per the verification pass) describe this same 229-column baseline as
  "trusted+untrusted" — which would imply it includes the 11 untrusted-derived columns. But a
  later markdown cell (cell 33) states plainly that "both arms use `X.columns` ('trusted', no
  untrusted features)" — directly contradicting cells 6/8, and matching what the code actually
  does. The code and cell 33 agree with each other; cells 6 and 8 are the odd ones out.
- **What I did:** wrote §2.7 to describe the baseline accurately as 229 columns (trusted +
  word_count, no untrusted columns), matching the code and cell 33, and added an inline forward
  reference to this note rather than silently adopting either the 06-markdown's mislabeling or the
  04.1/04.2 feature set. I did not edit `06_strategy_ablation.ipynb` itself (per the task's
  restriction on modifying notebooks).
- **What I did not resolve, and need your input on:** *why* does `06_strategy_ablation.ipynb`
  rebuild its baseline from the 228/229-column trusted set rather than importing the exact
  239/240-column trusted+untrusted set that was actually selected and carried forward through
  `04.1`→`04.2`? Two readings are possible: (a) this was an intentional simplification — the
  untrusted features were a priori expected to add little, so the author chose a cleaner baseline
  for the ablation study specifically; or (b) this is an oversight — the intent was to carry the
  selected feature set forward, and the untrusted columns were dropped unintentionally when `06`'s
  input data was rebuilt from an earlier-stage results file rather than reconstructed from the
  04.1-selected matrix directly. I have no way to determine which from the notebooks alone. This
  matters for the Methods chapter's framing and potentially for Discussion/Limitations: if (b), it
  is worth a one-line acknowledgment that the ablation study's baseline is not a strict superset
  continuation of the originally selected model. If (a), no acknowledgment is needed beyond what
  §2.7 already states.

*(Superseded by the resolution above — kept only as the historical record of what was found and
why it mattered.)*

## 2. Other claims verified this session (for your awareness, not requiring action)

These were not previously independently re-verified at cell level in prior sessions and are now
confirmed directly:

- Exact classifier/resampler labels in `04.1`'s grid (`LogReg`/`RF`/`XGB` × `(none)`/`(under)`/
  `(over)`, where `(under)` = `RandomUnderSampler` and `(over)` = `SMOTE`) — rendered in the chapter
  as plain-language "logistic regression / random forest / XGBoost" and "no resampling / random
  undersampling / SMOTE oversampling" rather than the internal code labels, which is a stylistic
  simplification for readability, not a factual change.
- Inner validation split sizes for `04.1` (2,629 comments / 256 threads inner-train; 633 comments /
  64 threads validation) — confirmed via cell 12 output; not previously stated anywhere in
  `REPORT_OUTLINE.md`.
- The exact composition of the 41 handcrafted features (22 numeric + 6 match booleans + 3 scalars +
  10 one-hot tone columns) and the exact composition of the 11 untrusted-derived columns — confirmed
  via `04.1` cells 6–7 and `03.1`/`03.2` cells defining each feature. Not previously broken down to
  this level of detail in any planning document.
- `07_strategy_interpretation.ipynb`'s exact statistical methodology: Mann–Whitney U → rank-biserial
  r (`2U/(n1*n2) - 1`), Benjamini–Hochberg FDR correction **implemented locally in the notebook**
  (not via `statsmodels`, per the notebook's own docstring rationale), significance threshold 0.05,
  SHAP via `shap.TreeExplainer` on the fitted XGBoost step. Confirmed via cells 29–30 and 22–23.
- `05_rhetorical_scoring.ipynb`'s prompt-iteration rationale and the specific pre-/post-review
  human-agreement numbers for refutation and justification (16.7%→27.8% and 11.1%→22.2% respectively)
  — confirmed via cells 22, 33–37, 40.
- `08_wa_external_validation.ipynb` cell 19's protocol statement, quoted in the verification pass
  nearly in full — the chapter's §2.9 wording is a close paraphrase of that cell's own language,
  not a reconstruction from secondary documentation.
- WA corpus/split sizes: 8,106 collected / 8,096 scored (10 dropped), native split 6,584 train /
  1,512 heldout at the comment level. **One nuance not reflected in the chapter, flagged here**:
  cell 19's own markdown describes the same split at a different unit — "2,509 train / 542 heldout
  *conversations*" — which is a different count from the comment-level numbers because one
  conversation can contain multiple comments. The chapter uses the comment-level figures throughout
  (6,584/1,512) for consistency with how every other split size in this chapter is reported (in
  comments, not conversations/threads-as-units-at-the-OP-level). This is not a discrepancy, just
  two different valid units of count — flagged only so it isn't mistaken for an error if you
  cross-check against the notebook's own text.

## 3. Column-order orderings count — RESOLVED (2026-10-20): standardized on 8

`06`'s column-order sensitivity demonstration (cells 11–12) explicitly uses **six** orderings (the
original plus five random reshuffles) to produce its published 0.0422 swing figure. Directly
verified this session that `07_strategy_interpretation.ipynb` (cell 41/42: "every PR-AUC below is
an order-robust mean over 8 random column orderings, following `06`'s convention") and
`08_wa_external_validation.ipynb` (cells 20/32, same `n_repeats=8` default) both use **eight**
orderings as their standing default — confirming eight, not six, is the convention consistently
used everywhere else in the project, including for `06`'s own headline mean±SD table (cell 15's
`evaluate_stage_order_robust(..., n_repeats=8)`, used unchanged in cell 17).

At your request, first recomputed the specific sensitivity-demonstration swing with eight
orderings instead of six, using a standalone replication script (same as the untrusted-features
check above; original 237-column set, no change to that decision — this is purely about ordering
count). Eight orderings (original + seven reshuffles) gives a range of **0.0482** (0.4908 to
0.5390), slightly *larger* than the originally published six-ordering figure of 0.0422 (0.4968 to
0.5390) — the extra reshuffles found a marginally lower minimum. The chapter's underlying point
(column order alone causes a swing large enough to matter) is unaffected either way; if anything it
is marginally reinforced.

**You then asked for this fixed directly in the notebook, not just in the report.** Done
(2026-10-20): `06_strategy_ablation.ipynb` cell 12's code was extended from 5 to 7 random
reshuffles (8 orderings total, matching the `n_repeats=8` convention used everywhere else in this
and the following two notebooks), and the notebook was re-executed in place
(`jupyter nbconvert --execute --inplace`) so the new output is authentically generated, not
hand-written. Cell 13's markdown was updated from "swings PR-AUC by 0.04" to "swings PR-AUC by
roughly 0.05". Verified the full re-execution changed nothing else: every other PR-AUC/output in
the notebook, including cell 17's headline table (0.4467±0.0196 / 0.5202±0.0177 / 0.3538±0.0079),
reproduced bit-for-bit identical to before. `02_METHODS.md` §2.7 already matches this (updated in
the step above). **This item is now fully closed — notebook and report agree.**

**Downstream documents — propagated (2026-10-21).** At your request, updated every other text
reference to the old 0.0422 figure to 0.0482: `figures/FIGURE_NOTES.md`,
`FIGURE_SELECTION_PLAN.md`, `PRESENTATION_STORYLINE.md`, `RESULTS_SHORTLIST.md`,
`report_draft/INTRODUCTION_REVIEW_NOTES.md`, both Gamma presentation content files
(`presentation/GAMMA_CONTENT.md`, `GAMMA_CONTENT_V2.md`), and both Hebrew speaker-notes files
(`presentation/SPEAKER_NOTES_HE.md`, `SPEAKER_NOTES_HE_V2.md`). Every occurrence was a simple
numeric substitution with no surrounding wording that depended on the exact value. None of the
seven approved chart images (PNG/SVG) display this number on-chart, so no figure needed
regenerating. This item, and the ordering-count question as a whole, is now fully closed —
notebook, Methods chapter, and every presentation/planning document agree on 0.0482.
- The home dataset's test-fold no-skill baseline is reported in the chapter as "approximately 20.3%"
  (0.2031, from `04.1`/`04.2`'s own 709-comment test fold) and the rhetorical-scoring population's
  own no-skill baseline as 0.2382 (from `06`'s 2,753-row population's own test fold) — these are
  two different, correct numbers for two different populations, not a typo. `RESULTS_SHORTLIST.md`
  at one point headlines a table with "no-skill PR-AUC = 0.161," which is actually the whole-dataset
  prevalence rather than either test fold's own rate — I did not change that document (out of
  scope), but the Methods chapter deliberately uses the fold-specific rates throughout, consistent
  with the project's own stated principle of not conflating different populations' baselines.

## 4. Methodological limitations to carry into Discussion (not elaborated in Methods itself)

- LLM-scored rhetorical strategies have a real reliability ceiling, concentrated on exactly the two
  strategies (refutation, justification) that matter most for the project's headline interpretive
  findings.
- Most individual point-estimate comparisons in this project have overlapping bootstrap confidence
  intervals; only the word-count-discovery comparison (§2.4) and the strategy-ablation gain (§2.7)
  are described as clearing that bar decisively.
- XGBoost+SMOTE results on the 237-column feature matrix are sensitive to column order; this is
  disclosed as a methods-level fact in §2.7, but its implication for how much confidence to place in
  any single point estimate belongs more fully in Discussion.
- The WA external validation shows the *group-level* rhetorical-strategy signal recurring but not
  the *specific dominant strategy*; only an explicitly untested hypothesis (differing
  negative-sampling procedures) exists for why — stated in §2.9, to be revisited in Discussion.
- Single subreddit, single platform, English-only dataset; thread size structurally correlates with
  a thread's apparent convincing rate (noted in §2.1).
- `refutation_reannotation/` was not referenced anywhere in this chapter, consistent with its
  out-of-scope status. `wa_quick_experiments.ipynb` was referenced only to explain why its
  zero-shot finding is excluded from this report's scope, not to report any of its results.

## 5. Suggested figures and tables, with notebook sources

| Placeholder | Section | Source notebook(s) / cells |
|---|---|---|
| End-to-end pipeline diagram | §2.1 | Conceptual; mirrors `00_introduction.ipynb` §1.4 and the README's notebook-sequence table. Does not exist as a notebook figure. |
| Feature-family summary table | §2.2 | `03.1`/`03.2`/`03.3` (feature definitions); `04.1` cells 6–7 (counts). |
| Feature correlation heatmap | §2.2 | `03.3_feature_engineering_op_relation.ipynb`, final cell ("Feature Correlation Heatmap (w/o embedding features)") — exists as an embedded image; content not yet extracted. |
| Model-selection grid table | §2.3 | `04.1_model_training.ipynb` cell 13 (`validation_results` or equivalent DataFrame). |
| PR/ROC curve, selected pipeline | §2.3 | `04.1_model_training.ipynb` cell 17 — exists as an embedded image; content not yet extracted. |
| Word-count-by-outcome figure | §2.4 | `04.2_baseline_comparison.ipynb` cells 30–31. A presentation-ready version of this already exists as `degree_completion/figures/figure4_comment_length_by_outcome.png`; per the task's own scope note, decide separately whether to reuse it as-is or build a report-specific version. |
| Baseline comparison table | §2.4 | `04.2_baseline_comparison.ipynb` cells 9–10, 33, 77, 79. Partially covered by an existing chart, `degree_completion/figures/figure1_initial_model_comparison.png` — covers 3 of the 4 rows this table needs (engineered pipeline, TF-IDF, word-count-only; missing the word-count-augmented re-selection row, 0.4922), as a bar chart rather than a table. |
| Length-matched-sample comparison table | §2.5 | `04.3_length_controlled_analysis.ipynb` cells 19–22 (its own sample) and `06_strategy_ablation.ipynb` cells 31–35 (the other sample) — side by side, to visually reinforce that they are not the same construction. The `06` half is already covered by an existing chart, `degree_completion/figures/figure3_strategy_gain_length_matching.png` (full-vs-matched strategy-score gain, 84% retention); no equivalent chart exists for `04.3`'s half. |
| Eight-strategy rubric + reliability table | §2.6 | Definitions: `05_rhetorical_scoring.ipynb` cell 1. Reliability: cells 28–40. No existing figure covers this. |
| SHAP / rank-biserial comparison table | §2.8 | `07_strategy_interpretation.ipynb` cells 36 (SHAP), 39 (effect size). A presentation-ready SHAP comparison chart already exists as `degree_completion/figures/figure5_refutation_vs_justification_shap.png` — again, decide separately whether to reuse or rebuild for the report. |
| SHAP beeswarm plots | §2.8 | `07_strategy_interpretation.ipynb` cells 23 (full population), 35 (length-matched) — embedded images, not yet extracted; the project's own guidance (per `FIGURE_SELECTION_PLAN.md`) is that these may not be legible enough for print and a simplified comparison chart is preferable — `figure5_refutation_vs_justification_shap.png` (row above) is exactly that simplified alternative, already built. |
| Home-vs-WA construction comparison table | §2.9 | `02_data_annotation.ipynb` (home negative-sampling) vs. `08_wa_external_validation.ipynb` cells 5, 16, 18, 20 (WA corpus/split sizes) and its own markdown describing Jaccard-matched negatives. No existing figure covers this specific construction-comparison content. |

**Three approved figures not listed above because they show findings, not procedure, and belong in
Results rather than Methods:** `figure2_strategy_incremental_value.png` (the §2.7 ablation headline
gain), `figure6_strategy_ranking_cmv_vs_wa.png` (the §2.8/§2.9 cross-dataset strategy-ranking
reversal), and `figure7_wa_external_validation.png` (the §2.9 WA feature-set comparison). Flagged
here so it's clear they weren't missed — they're the right figures for Results, not omissions from
this table.

## 6. Methods-subsection → notebook/cell mapping

| Subsection | Primary notebook(s) | Key cells |
|---|---|---|
| 2.1 Data collection and annotation | `01_data_download.ipynb`, `02_data_annotation.ipynb` | `02` cell 15 (`thread_text` rationale); `02` cell 37 (final row count); cleanup cells for the 3 removed rows. |
| 2.2 Feature engineering | `03.1`, `03.2`, `03.3` | `03.1` cells 33, 58–59, 72, 90–91 (rejected→kept sequence, tone categories); `03.2` cells 15, 105–107, 116 (untrusted flags, embedding model, PCA); `03.3` cell 10 and later (relational diff/match features); `03.3` cell 127 (`03_untrusted_features.json`). |
| 2.3 Model training and evaluation | `04.1_model_training.ipynb` | Cells 6–8 (feature-set construction/counts), 8 (train/test split), 10 (grid definition), 12 (inner validation split), 13 (selection), 17 (test evaluation). |
| 2.4 Baseline investigation | `04.2_baseline_comparison.ipynb` | Cells 9–10 (baseline table), 30–31 (word-count histogram/stats), 33 (length_vs_others), 73–79 (re-selection with word_count, bootstrap CIs). |
| 2.5 Length-controlled analysis | `04.3_length_controlled_analysis.ipynb` | Cells 19–22 (lift tables, loose/strict samples), §8.11–8.12 per prior documentation (re-selection overfitting example). |
| 2.6 Rhetorical-strategy scoring | `05_rhetorical_scoring.ipynb` | Cell 1 (rubric/scale), 16/22 (v1→v2 prompt), 24/42 (population restriction, 2,753 rows), 28/31 (test-retest), 33–40 (human hand-scoring). |
| 2.7 Strategy ablation | `06_strategy_ablation.ipynb` | Cells 2–4 (population, baseline feature set — see §1 above), 9/17 (headline PR-AUC table), 11–14 (column-order sensitivity), 22–23/28–30 (9-feature combination, RF vs. XGB). |
| 2.8 Strategy interpretation | `07_strategy_interpretation.ipynb` | Cells 19–20 (model/feature-set used), 22–23 (SHAP setup), 29–30 (Mann–Whitney/FDR setup), 36/39 (comparison tables). |
| 2.9 External validation | `08_wa_external_validation.ipynb` | Cells 5 (corpus size), 16/18 (scoring, drops), 19 (protocol statement), 20 (split sizes), 39 (pairs assertion). |

## 7. Scope check

- No notebooks, experimental code, datasets, model outputs, or figures were modified or run.
- No existing report chapter, planning document, or the three research questions were modified.
- Results and Discussion were not started.
- `refutation_reannotation/` was not referenced. `wa_quick_experiments.ipynb` was referenced only to
  justify its exclusion, not to report its findings.
- All numeric claims are either previously `[VERIFIED]` in `REPORT_OUTLINE.md` or freshly
  cell-verified this session (see §2 above); none were invented or carried forward unverified.
