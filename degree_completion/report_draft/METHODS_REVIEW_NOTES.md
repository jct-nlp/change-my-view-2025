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

> **Addendum (2026-10-25), unrelated to this question — reopens nothing above.** The baseline
> numbers this resolution cites (0.4467/0.5202/+0.0735) are themselves now stale for an unrelated
> reason: cell 17's column-order-shuffle seed was later pinned to 42 (was 0) so it would match the
> Length-Controlled Benchmark section's own full-population numbers, which a presentation review
> caught disagreeing with these despite describing the same comparison. Current values: 0.4434 /
> 0.5098 / +0.0664. This resolution's own comparison (0.5202→0.5169, 0.4467→0.4858, the untrusted-
> columns replication) was never rerun under seed=42 and is left as-is, unchanged by this addendum —
> only the reference point shifted, not this section's closed finding.

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

> **Addendum (2026-10-25), unrelated to this question — reopens nothing above.** Cell 17's headline
> table quoted here as the bit-for-bit-reproduced value (0.4467±0.0196 / 0.5202±0.0177 /
> 0.3538±0.0079) was accurate as of this 2026-10-20 re-execution, but cell 17's three calls were
> later pinned to column-order-shuffle seed=42 (was 0) so this comparison would match the
> Length-Controlled Benchmark section's own full-population numbers below, which a presentation
> review caught disagreeing with these despite describing the same comparison. Current values:
> 0.4434±0.0187 / 0.5098±0.0200 / 0.3514±0.0065, gain +0.0664. `02_METHODS.md` §2.7 updated to
> match (2026-10-25). The 8-orderings-count fix this section documents is unaffected and still
> closed — this addendum only concerns which seed numbered those 8 orderings.

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
| End-to-end pipeline diagram | §2.1 | **Resolved 2026-10-26, simplified 2026-10-27 per your feedback:** built as `figures/figure8_pipeline_diagram.png` (presentation) / `figures/report/figure8_pipeline_diagram.png` (report), generated by `make_figures.py`'s `figure8()`. Conceptual schematic, not extracted from any notebook plot. Originally nine boxes (one per Methods subsection) with one highlighted in orange as the length-confound "pivot" — simplified to four uniform-colored, broader phases (data preparation 01–03.3; modeling & length confound 04.1–04.3; rhetorical-strategy analysis 05–07; external validation 08) after you flagged the nine-box version as over-detailed for a high-level figure and the orange highlight as blurring the Methods/Results distinction (a "finding" color on a "procedure" figure). Captioned and placed in `02_METHODS.md` §2.1. |
| Feature-family summary table | §2.2 | **Resolved 2026-10-27:** built as a 15-row Markdown table directly in `02_METHODS.md` §2.2. Every column count and the trusted/untrusted split is read directly from `04.1_model_training.ipynb` cell 6 (`numeric_features`, `categorical_features`, `match_features`, scalar features, `embedding_features`) and cell 7 (`untrusted_encoded`, sourced from `results/03_untrusted_features.json`: `sentiment_vader`, `tone_label`, `style_label`, `ethos_score`, `pathos_score`, `logos_score`). One-hot category counts (tone: 6 LLM categories → 10 cols for 2 sides; untrusted tone: 7 RoBERTa emotion categories → 6 cols; untrusted style: 2 BART-MNLI categories → 1 col) confirmed against the label definitions in `03.1_feature_engineering_part1.ipynb` (cells 53–59, 85–91). The 187-embedding-column count was independently re-verified by loading `results/03_cmv_comments_df.csv.zip` directly and counting `embedding_`-prefixed columns, not just taken from notebook prose. Row-by-row totals reconcile exactly to the already-stated 41 handcrafted / 11 untrusted / 228 trusted / 239 trusted+untrusted figures — no discrepancy found. |
| ~~Feature correlation heatmap~~ | §2.2 | **Cut 2026-10-27, by decision not oversight.** Its only stated job was to illustrate the feature-redundancy claim behind §2.7's column-order-sensitivity finding — but that claim is already established, more convincingly, by the empirical 0.0482 PR-AUC swing already stated in §2.7's prose; the heatmap would be illustrative, not load-bearing evidence. Weighed against that: a correlation matrix over 41 handcrafted features tends to render dense and hard to read in print, the same "more detail than the point needs" problem flagged for the original nine-box pipeline diagram. Nothing in `02_METHODS.md` cross-references it, so removing the placeholder breaks no link. The underlying embedded image in `03.3_feature_engineering_op_relation.ipynb`'s final cell was never opened/extracted — this was a scope decision made before looking at it, not a judgment on its content. |
| Model-selection grid table | §2.3 | **Resolved 2026-10-27:** built as a 27-row Markdown table in `02_METHODS.md` §2.3, pulled directly from `04.1_model_training.ipynb` cell 13's `val_df` output (not reconstructed or estimated). Classifier/resampling labels rendered in the same plain-language style already used elsewhere in the chapter (per §2 above). Explicitly flagged in the table's own caption that these are **validation**-PR-AUC scores used for selection, distinct from the 0.3848 **test**-PR-AUC already reported in the prose above the table (the selected row's own validation score is 0.3063) — this is the same validation-vs-test distinction the project is careful about elsewhere, so it needed stating here too rather than risking the two numbers being read as the same quantity. Also pulled cell 14's pivot table (per-model untrusted-feature impact) into one summary sentence after the table, including the `XGB (over)` +0.063 PR-AUC standout already referenced in `02_METHODS.md`'s own prose a few paragraphs earlier. |
| ~~PR/ROC curve, selected pipeline~~ | §2.3 | **Cut 2026-10-27, after opening and reviewing it (unlike the heatmap, this one's content was actually looked at before deciding).** `04.1_model_training.ipynb` cell 17's curves are for the test PR-AUC 0.3848 / ROC-AUC 0.6982 already stated in the §2.3 prose — a single model, no comparison series, so the curves trace the same two numbers already in the text rather than adding information. Weaker case than even the heatmap: this is also the PR-AUC of the pipeline the very next section (§2.4) immediately supersedes with simpler baselines, so it illustrates a result the chapter's own narrative moves past a page later. The one argument for keeping it — PR/ROC curves as a conventional "shows evaluation rigor" artifact — was judged weaker than the cost of including a chart that adds no new information. Extracted image saved only to the local scratchpad for this review, not committed anywhere. |
| Word-count-by-outcome figure | §2.4 | `04.2_baseline_comparison.ipynb` cells 30–31. **Resolved 2026-10-26:** a report-specific version (title + fuller footer) exists at `degree_completion/figures/report/figure4_comment_length_by_outcome.png`, captioned and placed in `02_METHODS.md` §2.4. |
| Baseline comparison table | §2.4 | **Resolved 2026-10-27:** built as a 4-row table in `02_METHODS.md` §2.4, pulled directly from `04.2_baseline_comparison.ipynb` cells 9, 10, 33, and 77 output (not reconstructed). Includes PR-AUC, ROC-AUC, and F1 for all four models — Figure 1 only shows PR-AUC as a bar chart, so this table is the first place ROC-AUC/F1 for these four appear in the report. Deliberately does **not** repeat the order-robust mean±SD / bootstrap-CI numbers for the first and last rows (cell 79) since those are already stated in the prose immediately above the table — would have been pure duplication. Also pulled cell 9's majority-class (0.2031 PR-AUC) and stratified-random (0.2013) dummy-classifier rows as the single no-skill reference cited in the table caption rather than as separate table rows, since §2.3 already establishes the no-skill-equals-positive-rate convention generally and repeating two near-identical dummy rows would add little. |
| Length-matched-sample comparison table | §2.5 | **Resolved 2026-10-27:** built as a 3-row table (not 2 — added `04.3`'s loose-floor variant alongside its strict match, since both are discussed in the surrounding prose and a reader comparing numbers across the chapter benefits from one complete reference) in `02_METHODS.md` §2.5. All values pulled directly from live cell output: `04.3_length_controlled_analysis.ipynb` cell 19 (loose floor: 1,638 rows, r=0.0930; strict match: 1,235 rows, r=0.0655; both against the full dataset's r=0.3716) and `06_strategy_ablation.ipynb` cell 32 (its own decile-bin match: 1,235 rows, r=0.0608, against the scored population's own r=0.2882 — not the full-dataset r, a different unmatched baseline, correctly kept separate in the table). The table's own caption leads with the exact n=1,235 coincidence between `04.3`'s strict match and `06`'s sample, since that's precisely the confusion this project's documentation has repeatedly flagged elsewhere (`RESULTS_SHORTLIST.md` Finding 3, `FIGURE_SELECTION_PLAN.md`) — the "Used to answer" column is what actually disambiguates them. |
| Eight-strategy rubric + reliability table | §2.6 | **Resolved 2026-10-27:** built as an 8-row table in `02_METHODS.md` §2.6. Definitions condensed from `05_rhetorical_scoring.ipynb` cell 1's rubric (one-liner only, dropped the 1/5 scale anchors). Test-retest column uses `within_1_pct` from cell 30 (n=200) — the same metric the surrounding prose already cites ("95–99% within one point"), not `exact_match_pct`, to stay consistent with how the prose characterizes that check. Human-agreement column uses `exact_match_pct` before → after from cell 37 (n=18) — again matching the prose's own chosen metric (16.7%→27.8%, 11.1%→22.2% for refutation/justification specifically). All values pulled directly from cell output, not re-derived. One thing worth flagging: `call_to_action`'s exact-match percentage is identical before and after review (33.3%) even though 3 corrections were applied to it per cell 37's `corrections` dict — its mean absolute difference still improved (0.83→0.67), so the corrections moved near-misses closer without creating new exact matches. Reported as-is, not smoothed over. `personal_narrative` and `question_asking` have no correction entries in cell 37 at all, hence identical before/after numbers — also reported as-is. The after-review refutation/justification MAD figures (0.89 both) independently confirm the "~0.89, highest of the eight" claim already in the chapter's prose. |
| SHAP / rank-biserial comparison table | §2.8 | **Resolved 2026-10-27:** built as an 8-row table in `02_METHODS.md` §2.8, pulled directly from `07_strategy_interpretation.ipynb` cell 36 (`shap_comparison`) and cell 39 (`effect_size_comparison`) output. SHAP values matched the already-verified `FIGURE_NOTES.md` Figure 5 numbers exactly (independent cross-check, no discrepancy). The length-matched rank-biserial column and its FDR-significance flags were new extraction — not previously pulled into any planning document at the per-strategy level. One flagged result: `personal_narrative`'s rank-biserial effect is non-significant on the full population but significant *and negative* on the length-matched sample (r = −0.0859) — the only strategy whose significance and sign both flip. This is not a new finding: `FIGURE_SELECTION_PLAN.md`, `PRESENTATION_STORYLINE.md`, and `RESULTS_SHORTLIST.md` already document this exact number and call it out ("personal_narrative flips to r=−0.086" / "gains significance under matching") — this table just independently reconfirms it from the live cell, consistent with the existing record. |
| ~~SHAP beeswarm plots~~ | §2.8 | **Cut 2026-10-27, after extracting and actually looking at both (full population, cell 23; length-matched, cell 35) — contrary to `FIGURE_SELECTION_PLAN.md`'s untested prediction, both render legibly at full resolution, not illegibly dense.** This was a genuine judgment call, not a clean cut like the heatmap/PR-ROC curve: the beeswarms show *direction* of effect (color = feature value), information the magnitude table (row above) and Figure 5 don't carry. Flagged to you directly rather than decided unilaterally, since the case for inclusion was real. **You chose to cut both.** No placeholder existed in `02_METHODS.md` itself to remove — Figure 5 had already taken that slot in an earlier session; this row only needed closing out here. Extracted images saved only to the local scratchpad for this review, not committed anywhere. |
| Home-vs-WA construction comparison table | §2.9 | **Resolved 2026-10-27:** built as a 9-row comparison table in `02_METHODS.md` §2.9. WA-side numbers pulled directly from `08_wa_external_validation.ipynb` cell 5 (8,106 comments / 3,051 posts, pre-scoring), cell 18 (10 dropped, 6 train + 4 heldout, for unparseable Gemini responses — not previously broken out by split in any prior document), and cell 20 (6,584 train / 1,512 heldout, no-skill PR-AUC 0.5291, 4,263 pairs / 805 heldout). Jaccard-matching description reused from cell 1's markdown (already read this session). Home-side numbers reused from already-`[VERIFIED]` §2.1 facts, not re-extracted. All figures cross-check cleanly against the chapter's existing §2.9 prose — no discrepancies found. |

**Section 5 is now fully resolved** — every row above is either built (8 tables/figures) or cut by explicit decision (correlation heatmap, PR/ROC curve, SHAP beeswarms), each with its reasoning logged inline. Nothing in this section remains outstanding.

**Four approved figures not listed above because they show findings, not procedure, and belong in
Results rather than Methods:** `figure2_strategy_incremental_value.png` (the §2.7 ablation headline
gain), `figure6_strategy_ranking_cmv_vs_wa.png` (the §2.8/§2.9 cross-dataset strategy-ranking
reversal, rank-biserial), `figure9_strategy_ranking_shap_cmv_vs_wa.png` (the same cross-dataset
comparison, SHAP side — added 2026-10-28), and `figure7_wa_external_validation.png` (the §2.9 WA
feature-set comparison). Flagged here so it's clear they weren't missed — they're the right
figures for Results, not omissions from this table.

## 5a. Figure caption numbering — deferred, not yet assigned

**2026-10-26, updated same day after Figure 8 (pipeline diagram) was built:** Captions for the five
approved figures placed in `02_METHODS.md` (`figure8`/`figure1`/`figure4`/`figure3`/`figure5`-named
files, in that order of appearance) and the three placed as staged captions in
`RESULTS_SHORTLIST.md` (`figure2`/`figure7`/`figure6`) all use `**Figure N.**` rather than a
resolved number. This is deliberate, not an oversight: two other figure/table placeholders in
`02_METHODS.md` (feature correlation heatmap §2.2, PR/ROC curve §2.3) still sit *before* some of
the captioned figures and are not yet built, and the Results chapter itself has not been drafted.
Any number assigned now would likely be wrong once those are resolved. **Assign final sequential
`Figure N` numbers in one pass, by order of actual appearance in the finished document, only after
every figure (built and not-yet-built) and the Results chapter are in place** — not incrementally
as each is drafted.

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

## 8. Inner-validation no-skill baseline — RESOLVED (2026-10-28): corrected to the fold's own rate

§2.3's paragraph after the 27-combination grid stated "every combination beats the validation
fold's own no-skill PR-AUC baseline of 0.161," but that 0.161 was actually the **whole-dataset**
positive rate used as a stand-in — the validation fold's own rate had never been separately
checked, a gap the sentence itself flagged. You asked to check it rather than leave it as a known
gap.

**What was done:** reproduced the inner-validation split read-only, in a standalone script — same
data load, same `thread_id` hashing, same two `GroupShuffleSplit` calls (`random_state=42` for the
outer train/test split, `random_state=0` for the inner train/validation split) as
`04.1_model_training.ipynb` cells 4, 8, and 12, with no model fit and nothing written back. The
reproduction's split sizes (633 validation / 2,629 inner-train) matched the notebook's own printed
output exactly, confirming the same split was reconstructed, not a different one.

**Result:** the validation fold's own positive rate is **0.1690**, not 0.1614 (whole-dataset) and
not the previously-stated 0.161. This is not just a more precise version of the same claim — it
changes which claim is true. The grid's weakest row, `RF (none) | embeddings-only` at PR-AUC
0.1506, is below **both** 0.1690 and 0.1614, so "every combination beats no-skill" was already
false under the original (wrong) baseline too, not only under the corrected one. `02_METHODS.md`
§2.3 now states the correct fold-specific baseline (0.1690) and the accurate claim: 26 of 27
combinations beat it, with `RF (none) | embeddings-only` named as the sole exception that fails
under either baseline. This item is now closed.

## 9. Figure footers — stripped verification dates and redundant file pointers (2026-10-28)

Every figure's report-mode footer (`make_figures.py`, `verbose_footer` branch) carried a
"Verified YYYY-MM-DD" stamp — QA tracking useful during drafting, meaningless to a reader of the
finished report. Removed from all nine figures (1–9). Six of them also pointed to "Full caveats in
`FIGURE_NOTES.md`" — an internal project-tracking file a report reader can't follow. Removed that
pointer from Figures 1, 3, 4, 5 (already placed in `02_METHODS.md`, where the real caveats now
live in the chapter's own captions — the pointer was dead weight). Kept on Figures 2, 6, 7, which
are still staged for the not-yet-drafted Results chapter and currently have no other home for
those caveats; drop the pointer from each once its caveats are folded into Results prose the same
way Methods absorbed 1/3/4/5's. All nine report-mode PNG/SVGs regenerated; no numeric values
changed, footer text only.
