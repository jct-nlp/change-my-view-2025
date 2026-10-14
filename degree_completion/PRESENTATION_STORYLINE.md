# Presentation Storyline — MSc Defense (15 minutes)

**Status:** Revised planning document, 14 October 2026. Follows the advisor's ten-part structure,
the three research questions, and a timed 14:30-minute plan, with detailed speaker-content guidance
restored under each section so this document can serve as the main reference for writing slides and
speaker notes. **Not slides.** Times are targets to validate in rehearsal.

**Hard limit:** 15:00. **Planned speaking time:** 14:30, leaving 0:30 for transitions. Defense:
4 November 2026.

**Audience assumption:** understands machine learning (metrics, classifiers, confounds, train/test
splits) but has likely never heard of ChangeMyView or seen a "delta." Give every CMV-specific term
one plain-language sentence before using it.

## Three research questions (advisor's framing)

1. How well can textual features predict persuasive success, and what role does comment length play?
2. Do rhetorical strategies add predictive information beyond length and other textual features?
3. Which rhetorical strategies contribute to prediction, and how stable are the findings across
   datasets constructed differently?

**Narrative:** Initial prediction → discovery of the length confound → testing signal beyond
length → identifying specific strategies → independent-corpus validation and its limits.

## Slide plan and timing

The ten advisor sections below are **narrative sections**, not a cap of ten physical slides. One
main message per slide. A few sections may need two visual slides, but keep the total duration.

| Section | Time window | Duration | Main message / visual | RQ |
|---|---|---:|---|---|
| 1. Problem, motivation, research questions | 0:00–1:20 | 1:20 | Introduce CMV and the delta in ~30–40 seconds; explain the prediction task and the three questions. Minimal schematic or example. | All |
| 2. Dataset construction and task definition | 1:20–2:45 | 1:25 | 3,971 comments, 641 delta-awarded (16.1%); labels, negatives, thread-disjoint split; PR-AUC. Small pipeline diagram. | RQ1 |
| 3. Initial experiments and the length finding | 2:45–4:40 | 1:55 | Engineered PR-AUC 0.3848 vs TF-IDF 0.5119 vs word-count-only 0.5256; length histogram. Make clear these numbers are from the same original held-out test. | RQ1 |
| 4. Central question: beyond length? | 4:40–5:10 | 0:30 | Explicit transition: predictive association with length is not sufficient to establish rhetorical signal. Pose the next test. | RQ2 |
| 5. Eight strategies and measurement | 5:10–6:20 | 1:10 | One compact overview of all eight, 1–5 LLM scoring. Give most attention to **refutation** and **justification** as examples; do not spend equal time defining all eight. Mention small human check and limitations briefly, revisit near interpretation. | RQ2–3 |
| 6. Strategy-combination results | 6:20–7:55 | 1:35 | PR-AUC 0.4467±0.0196 → 0.5202±0.0177 after adding eight scores; column-order sensitivity handled by reporting mean±SD. Simple bar/dot chart. | RQ2 |
| 7. Length matching and compact model | 7:55–9:45 | 1:50 | Strategy-gain retention ~84% after length matching; nine features (eight scores + word count) PR-AUC 0.5379±0.0109 vs full 237 features 0.5202±0.0177. Avoid implying causal independence. | RQ2 |
| 8. Refutation vs justification | 9:45–11:10 | 1:25 | Home data: refutation remains prominent after length control; justification loses much of its importance. Simplified before/after SHAP ranking, not two dense beeswarms unless readable. Mention that both are among the least reliable vs human scoring. | RQ3 |
| 9. WA external validation and surprise | 11:10–13:10 | 2:00 | **Same preselected model architecture/configuration, refitted on WA training split and evaluated on WA held-out data; no new model-family/resampling selection on WA. NOT zero-shot.** Strategy signal persists; justification dominates instead of refutation. Sampling explanation is a hypothesis only. | RQ3 |
| 10. Conclusions, contribution, limitations | 13:10–14:30 | 1:20 | Three answers to RQ1–RQ3; predictive not causal; delta proxy; measurement reliability; dataset/sampling differences; future test of sampling hypothesis. | All |

**Total:** 14:30 + 0:30 margin. Do not schedule Q&A within the 15-minute presentation unless the
department explicitly requires it.

---

## Section-by-section speaker content

### 1. Problem, motivation, research questions (0:00–1:20, all RQs)

**Central message:** give the audience just enough of CMV's mechanics to follow everything after,
then state the three questions that organize the whole talk.

**Sequence of arguments:**
- r/changemyview is a public debate forum: someone posts an opinion they're open to changing,
  others reply trying to change it.
- The "delta" (∆) is the forum's own built-in success label: if the original poster is genuinely
  persuaded by a specific reply, they award that reply a delta. This is a mechanism the community
  itself has used for years, not something this project invented or inferred — and it gives a free,
  per-*comment* label for "this specific reply was persuasive," not just a vague thread outcome.
- Bridge sentence: "we treat predicting which comments get a delta as the core machine-learning
  problem."
- State the three research questions (see above) as the organizing frame for the rest of the talk.
- One sentence of motivation only — automated debate analysis, content moderation, persuasive-
  writing assistance — without dwelling; the audience's attention should go to method and findings.

**Suggested visual:** a simple annotated mockup of a post plus a delta-awarded reply (no such image
exists in any notebook; build fresh). Keep this the one slide that is pure explanation.

**Source notebook:** `this work/00_introduction.ipynb` §§1.1–1.3.

**Caveats:** keep the plain-language framing tight — one sentence per CMV-specific term before
using it, then move on.

---

### 2. Dataset construction and task definition (1:20–2:45, RQ1)

**Central message:** a concrete, per-comment supervised task, built with a leakage-free evaluation
discipline.

**Sequence of arguments:**
- Source: 500 top r/changemyview posts via the Reddit API (PRAW), adapted from Priniski & Horne's
  own collection code, parsed into one row per reply (comment).
- Final dataset after a documented cleanup (3 rows removed: edited-to-placeholder text, a bot
  notice mistaken for a comment, a numeric-grid body): **3,971 comments, 641 delta-awarded
  (16.1%), 3,330 not** — verified directly from `results/02_comments_df_base.csv.zip`.
- `thread_text` is defined as only the same commenter's own prior turns in the thread, not the full
  thread — an explicit design choice trading full-thread context for consistent context
  style/sentiment, at the acknowledged cost of missing same-side input from other commenters.
- Negatives are simply the non-delta-awarded replies already in the same threads, with no separate
  similarity filtering — worth flagging here briefly, since it matters later when contrasting with
  WA's differently-constructed negatives (Section 9).
- Evaluation: thread-grouped `GroupShuffleSplit` (80/20 by thread), zero thread overlap between
  train and test, verified by assertion — prevents the model from "peeking" at a related comment
  from the same discussion.
- Metric: PR-AUC chosen over accuracy/ROC-AUC because only ~16% of comments are positive.

**Suggested visual:** a small pipeline diagram — data collection → annotation → feature engineering
→ modeling → strategy scoring → external validation (does not exist as a figure in any notebook;
build fresh).

**Source notebooks:** `00_introduction.ipynb` §1.4; `01_data_download.ipynb`;
`02_data_annotation.ipynb`; `04.1_model_training.ipynb` (split/metric choice).

**Caveats:** verify the exact sampling/negative-construction wording directly against
`02_data_annotation.ipynb` before finalizing slide text.

---

### 3. Initial experiments and the length finding (2:45–4:40, RQ1)

**Central message:** the project's central methodological pivot — a bare-minimum baseline beat the
carefully engineered model, and diagnosing why surfaced a length confound.

**Sequence of arguments:**
1. Full engineered-feature pipeline (237 features, including 6 deliberately-kept "untrusted" ones;
   XGBoost + SMOTE, selected via a 3 classifiers × 3 resamplers × 3 feature-variants = 27-combination
   grid): test PR-AUC **0.3848**.
2. A bare-minimum TF-IDF + Logistic Regression baseline, using none of the engineered features:
   PR-AUC **0.5119** — beats the full pipeline.
3. Diagnose why: the TF-IDF model's top predictive tokens are mostly stopwords and the token
   "edit" — a sign it is tracking comment *length*, not content.
4. Test directly: `word_count` alone (one feature, Logistic Regression): PR-AUC **0.5256** —
   the best score yet. Mean word count 78.0 (non-convincing) vs 216.7 (convincing); r=0.3716
   (p=2.92e-130). Within-thread check, which rules out a pure topic/thread-size artifact: in 200
   threads containing both classes, the convincing comment was longer than that thread's own
   non-convincing average in 153/200 (76.5%, Wilcoxon p=4.15e-19).
5. The fix: fold `word_count` into the engineered feature set and re-run model selection from
   scratch. This closes most of the gap: test PR-AUC **0.4922**, with bootstrap 95% CIs barely
   overlapping the original selection — the single most statistically decisive comparison in the
   project.

**Framing line:** "this wasn't a failure — catching it is what let the rest of the project ask a
sharper question: is there anything beyond length that matters?"

**Suggested visual:** the `word_count`-by-outcome histogram (binwidth 50, comments <600 words) —
exists only as embedded notebook output; re-export before use.

**Source notebooks:** `04.1_model_training.ipynb` (original pipeline numbers); `04.2_baseline_
comparison.ipynb` (TF-IDF baseline, word-count discovery, re-selection with word_count).

**Caveats:** most point-estimate PR-AUC comparisons in this project have overlapping bootstrap 95%
CIs at this test-set size (709 comments / 81 threads); the word-count-vs-engineered gap is the one
comparison here with CIs that barely overlap — treat it as the most, not the only, solid point.

---

### 4. Central question: beyond length? (4:40–5:10, RQ2)

**Central message:** an explicit, short transition — showing a model does better *because of*
length is not the same as showing rhetorical content itself matters.

**Argument:** predictive association with length alone cannot establish a rhetorical signal; the
next test has to isolate what survives once length is controlled for. This poses RQ2/RQ3 directly:
do explicitly defined rhetorical strategies add predictive information beyond length (RQ2), and if
so, which ones, and how stably (RQ3)?

**Suggested visual:** none required — a text-only transition, or merge visually into the opening of
Section 5.

---

### 5. Eight strategies and measurement (5:10–6:20, RQ2–3)

**Central message:** one compact overview of the measurement instrument — keep it brief; detailed
interpretation is reserved for refutation and justification later.

**Sequence of arguments:**
- Eight strategies, each scored 1–5 by an LLM (Gemini), using a method adapted from Labruna et al.
  (2026)'s LLM-based persuasion-strategy scoring (originally 6 propaganda techniques on news text;
  adapted here to 8 new, CMV-specific constructive strategies): `justification`, `concession`,
  `evidence`, `refutation`, `personal_narrative`, `call_to_action`, `simplification`,
  `question_asking`.
- Give a one-line definition only for the two that matter most downstream — **refutation**:
  directly countering a specific flaw in the original argument; **justification**: giving a
  reason/support beyond bare assertion. Do not spend equal time defining all eight on screen.
- Scoring is restricted to comments ≥30 words (or any length if convincing): 2,753 of 3,971
  comments. Flag this denominator shift briefly if citing any percentage from this stage onward
  (e.g. a 23.3% convincing rate here is not directly comparable to the 16.1% baseline from Section 2).
- Validation, mentioned briefly here and revisited as a direct caveat on the Section 8
  interpretation: test-retest self-consistency is strong (95–99% within-1-point agreement across
  all 8 strategies); agreement with a single human judge (N=18) is weaker overall, and weakest
  specifically on refutation and justification. State plainly that this is a lightweight check, not
  a formal inter-annotator reliability study.

**Suggested visual:** none essential; a compact 8-strategy label row if a visual anchor is wanted.

**Source notebook:** `05_rhetorical_scoring.ipynb`.

**Caveats:** strategy scores are LLM-generated measurements on a rubric, not direct observations of
a mental state or definitive human annotations.

---

### 6. Strategy-combination results (6:20–7:55, RQ2)

**Central message:** adding the eight strategy scores produces a real, order-robust predictive gain
beyond the existing engineered model.

**Sequence of arguments:**
- Engineered model alone (full population): PR-AUC **0.4467±0.0196**.
- + 8 strategy scores: PR-AUC **0.5202±0.0177** (+0.0735) — a gain that clears the column-order
  noise band described below.
- Strategy scores alone: PR-AUC **0.3538±0.0079** — real standalone signal, between no-skill
  (0.2382) and the full model.

**Methodological disclosure (state this, don't footnote it):** the underlying XGBoost+SMOTE
pipeline, on this correlated 200+-column feature matrix, is sensitive to column *order* —
reshuffling only column order swings PR-AUC by as much as 0.0422. Root-caused to XGBoost's
deterministic tie-breaking among near-equal split gains on correlated features (not a SMOTE
artifact). Every PR-AUC number from this point on in the talk is reported as mean±SD over multiple
column orderings for exactly this reason.

**Suggested visual:** a simple bar/dot chart (engineered-alone vs +strategies vs strategies-alone)
with error bars.

**Source notebook:** `06_strategy_ablation.ipynb`.

**Caveats:** always quote mean±SD for these numbers; never present a single run's value.

---

### 7. Length matching and compact model (7:55–9:45, RQ2)

**Central message:** the strategy-score gain survives length control far better than the
engineered-feature family it's measured against.

**Two separate length-matched experiments exist in this project and must not be shown as one
unified comparison** — different source notebooks, different populations, different purposes:

- `04.3_length_controlled_analysis.ipynb` matches **within the full 3,971-row dataset** (loose
  floor: 1,638 comments; strict 1:1 match: 1,235 comments) and re-measures each of three
  *different* models' (`TF-IDF`, `word_count` alone, the engineered pipeline) own lift-over-no-skill,
  before vs. after matching.
- `06_strategy_ablation.ipynb` matches separately, **within its own 2,753-row LLM-scored subset**
  (strict 1:1 match: 1,235 comments — a coincidentally similar row count, not the same sample: `06`
  cell 31 explains explicitly why `04.3`'s sample can't be reused here, since many of its matched
  non-convincing rows fall under the 30-word scoring floor and have no strategy score), and asks a
  narrower, different question: does the *incremental gain from adding the 8 strategy scores*
  (not any single model's own lift) survive the same kind of matching.

**For the main length-matching slide, present only `06`'s self-contained result:** the
strategy-score gain itself is **0.0664 PR-AUC** on the full scored population (n=2,753) vs.
**0.0557** on its own length-matched sample (n=1,235) — **84% retention** (`06` cells 32/35). This
is the slide's only number from this section; do not add `04.3`'s word-count/engineered-pipeline
retention figures to the same chart or caption as if they were a directly comparable, same-sample
measurement.

**If `04.3`'s result is mentioned at all** (e.g. as a verbal aside, not on the main chart), state it
as a separate check on a separate sample: `04.3` §8.11 found `word_count`'s own lift-over-no-skill
fell from 0.3225 (full dataset) to 0.0576 (its own strict 1:1 match) — **~18% retained** — and the
engineered pipeline's own lift fell from 0.1817 to 0.0334, **also ~18% retained** (0.0334/0.1817 =
18.4%), not 25%. **Resolved 2026-10-14:** `06`'s own markdown cell 36 cites "~25%" for the engineered
pipeline's retention and attributes it to `04.3`, but that is a mix-up within `06`'s narrative text —
the only "25%" figure anywhere in `04.3` is a different quantity on a different sample: the
*residual `word_count` correlation* remaining after `04.3`'s *loose* 84-word floor
(r=0.093/0.3716=25.0%, `04.3` cell 41), not the engineered pipeline's lift retention under strict
matching. `04.3`'s own cells 21–22 are internally consistent and give ~18% for both `word_count` and
the engineered pipeline. **Use ~18% for both, never 25%, on any slide, in the report, or in Q&A.**
(The research computation itself is correct and unchanged — only `06`'s one-sentence description of
it is wrong; per `RESEARCH_GUARDRAILS.md`'s standing rule to flag rather than edit research
notebooks, `06_strategy_ablation.ipynb` cell 36 has been left as-is. Decide separately if you want
that notebook cell's prose corrected too.)
- Best single feature combination found anywhere in the project: the 8 strategy scores +
  `word_count` alone (9 columns) = PR-AUC **0.5379±0.0109** (XGBoost, `06` cell 22) — beats every
  larger combination tested, including the full 237-column set. Useful as a parsimony/
  interpretability counterpoint if a "why not just throw in everything" question comes up.
- **XGBoost vs. Random Forest on this 9-column set — do not call XGBoost "the best architecture"
  here.** `06` cells 28–30 re-ran the full classifier×resampling grid on just these 9 columns:
  `RF (over)` edges out `XGB (over)` on both validation (0.4331 vs. 0.4133) and the order-robust
  held-out test comparison (0.5432±0.0065 vs. 0.5380±0.0108) — a 0.0052 gap, smaller than either
  architecture's own order-induced spread, so not a confident win either way. The notebook's own
  conclusion, and this project's, is to keep `XGB (over)` for consistency with every other result in
  the talk, not because it was shown to outperform Random Forest on this feature set — say "XGBoost
  (0.5380), with Random Forest scoring statistically indistinguishably at 0.5432" if asked, never
  "XGBoost was the best architecture for this combination."
- Optional aside if time allows: re-running the full 27-combination selection grid from scratch on
  the small 1,235-row matched sample (`04.3` §8.12) picked a *different* architecture that scored
  *below* its own no-skill baseline on test (lift −0.0162) — a small-validation-set overfitting case
  (205 validation rows, 27 candidates) — while simply reusing the architecture already selected on
  the full dataset scored positively (lift +0.0334). A good "lessons learned" aside, not essential to
  the core arc, and unrelated to the 9-feature XGB-vs-RF question above (different sample, different
  feature set).

**Suggested visual:** the strategy-score-gain retention chart from `06` alone (full population vs.
its own length-matched sample) — two bars, not three populations merged. Avoid implying causal
independence from length — frame as "the strategy-score gain survives length matching much better
than the feature families it's layered on top of."

**Source notebooks:** `06_strategy_ablation.ipynb` (primary, for the main chart);
`04.3_length_controlled_analysis.ipynb` (background only, cite as a separate check if mentioned).

**Caveats:** length matching changes the sample and reduces power; state which notebook, which
population, and which matching protocol any retention percentage is computed on — never leave it
implicit which of the two differently-built "length-matched sample" constructions a number refers
to.

---

### 8. Refutation vs justification (9:45–11:10, RQ3)

**Central message:** on the home dataset, `refutation` holds up most consistently under length
control; `justification`'s apparent importance is mostly a length proxy.

**Sequence of arguments:**
- SHAP mean |value|, full population: `word_count` 1.0245 » **refutation 0.6034** » justification
  0.4156 > evidence 0.3592 > concession 0.3439 > simplification 0.3193 > personal_narrative 0.1853
  ≈ call_to_action 0.1837 > question_asking 0.1518 (least influential).
- SHAP mean |value|, strict length-matched: `word_count` 0.8402 » **refutation 0.5193** (holds
  rank, modest decline) > concession 0.3445 > simplification 0.3383 > evidence 0.3090 >
  call_to_action 0.2707 > personal_narrative 0.2168 > **justification 0.1956** (collapses from 3rd
  to 7th-of-8 — most of its full-population importance was length-related) > question_asking 0.1806.
- Cross-checked with a model-free Mann-Whitney rank-biserial effect size (FDR-corrected): full
  population justification r=0.3584 > refutation r=0.3149 > simplification r=0.1944 >
  call_to_action r=0.1584 > concession r=0.1481 > evidence r=0.1018 > question_asking r=0.0069
  (n.s.) > personal_narrative r=−0.0217 (n.s.). Length-matched, only 3 of 8 strategies remain
  significant: refutation (r=0.154, 49% retained), justification (r=0.131, 37% retained),
  simplification (r=0.079). `personal_narrative` is the one strategy that *gains* significance
  under matching, flipping to a small, reliable *negative* association (r=−0.086).

**Caveat to state directly next to this finding, not filed separately:** refutation and
justification are also the two strategies `05_rhetorical_scoring.ipynb`'s own validation flagged as
hardest for the LLM to calibrate against human judgment (lowest exact-match agreement in both the
test-retest and hand-scoring checks). The strongest univariate signal in this analysis arm comes
from the two most noisily-scored strategies.

**Suggested visual:** the two SHAP rankings shown side by side (full population vs length-matched)
— the clearest single visual argument in the project for "refutation holds up, justification
doesn't." Simplify into a before/after ranking rather than two dense beeswarms unless both are
readable on a projector.

**Source notebook:** `07_strategy_interpretation.ipynb` (SHAP/effect sizes); `05_rhetorical_
scoring.ipynb` (reliability caveat).

**Caveats:** SHAP importance and univariate effect size measure different things; do not call
either causal influence.

---

### 9. WA external validation and surprise (11:10–13:10, RQ3)

**Central message:** the structural finding replicates on an independent corpus; the specific
dominant strategy does not — an honestly reported partial replication.

**Protocol — state this precisely, it is the corrected wording and must not be garbled in
delivery:**
- The model architecture and training configuration already selected on the home dataset
  (`RobustScaler` + `SMOTE` + `XGBoost`, chosen in `04.1`/`04.2`, reused unchanged through `06`/
  `07`) is carried over as-is. **No new architecture/configuration selection is performed on WA.**
- A **new model instance is then fit on WA's own native training split and evaluated on WA's own
  native heldout split** (verified directly against `08_wa_external_validation.ipynb` cell 19).
- This is reuse of a selected configuration plus a refit on WA's own data — distinct from
  (a) selecting/tuning a *new* configuration using WA data (not done), and (b) zero-shot transfer
  without refitting (not done here, and out of scope for this presentation — see below). **Do not
  call this zero-shot transfer or a frozen model.**

**Quantitative results (WA corpus: 8,106 comments, no-skill baseline 0.5291; word-count confound
weaker here, r=0.1516 vs 0.3716 at home):**
- `word_count` only: PR-AUC 0.5767 (lift 0.0476).
- strategy scores only: PR-AUC 0.5967 (lift 0.0676).
- robust 4-feature core (refutation, justification, simplification, word_count): PR-AUC 0.5919
  (lift 0.0628); pairwise accuracy 58.68% — above the literature's "Independent Scoring" baselines
  (53.78–56.66%, Labruna et al. 2026) and below their full MS-PS method (60.59–64.53%) — a
  reasonable, honestly bounded place to land, not claimed as a like-for-like beat of the
  literature.

**The reversal, stated precisely:** on WA, `justification` is the top strategy by both SHAP and
Mann-Whitney effect size (r=0.1962), while `refutation` is 4th by effect size and near-bottom by
SHAP — the opposite ranking from home. Length control on WA still leaves `justification`'s effect
84% intact (unlike at home, where it collapsed). WA's own test-retest reliability for every
strategy (including refutation, 76.5% exact match) is better than at home, which rules out
"scoring noise" as the explanation.

**Hypothesis — explicitly labeled as untested, not a conclusion:** the two datasets build their
negative examples differently — WA's losing replies are Jaccard-matched to be topically similar
and substantive to the winner, while this project's non-convincing comments are not filtered for
similarity at all (36.6% of the home dataset's unfiltered non-convincing pool is under 30 words).
This is offered with one supporting, not proving, number; no experiment in this project tests it
directly.

**Suggested visual:** the WA feature-set comparison table (PR-AUC / pairwise accuracy) — more
presentation-friendly than another SHAP plot at this point in the talk; avoid two SHAP slides back
to back.

**Source notebook:** `08_wa_external_validation.ipynb`.

**Explicit exclusion:** exploratory `wa_quick_experiments.ipynb` work — the genuine zero-shot
(no-refit) check, the "embeddings added on top of the robust core" check, and the "re-adding
concession" check — is **out of scope for this slide, for every backup slide, and for the report**.
Do not reference any of it on screen. It may be mentioned informally only if directly asked, with
clear "exploratory, unreviewed" labeling.

---

### 10. Conclusions, contribution, limitations (13:10–14:30, all RQs)

**Central message:** three answers to RQ1–RQ3, read in decreasing order of confidence, framed as
predictive, not causal.

**Sequence of arguments:**
- **RQ1:** comment length is the single dominant predictive signal in this dataset — true,
  consistently, everywhere it was tested.
- **RQ2:** a real, length-independent rhetorical signal exists beyond length — also true, and it
  replicates structurally on an independent corpus.
- **RQ3:** that signal concentrates specifically in `refutation` on this project's own data — true
  here, but it did not replicate in that specific form on WA, where `justification` took its place;
  only an explicitly labeled, untested hypothesis exists for why.

**Limitations (state directly, do not bury):**
- Single subreddit, single platform, English only.
- LLM-scored strategy reliability is weakest on exactly the two strategies (`refutation`,
  `justification`) that matter most for the headline claims.
- Most individual point-estimate comparisons in this project have overlapping confidence intervals
  at this sample size; the comparisons strong enough to call decisive are the ones already
  highlighted (the word-count discovery in Section 3, the strategy-ablation gain in Section 6).
- A delta is an observable CMV success label, not a complete or error-free measurement of internal
  belief change.

**Suggested future work (explicitly future, not completed):** directly test the negative-sampling
hypothesis from Section 9; extend validation to a third, differently-constructed corpus.

**Suggested visual:** none required; a condensed version of the RQ table can anchor this slide.

**Source:** synthesis across all notebooks; see `REPORT_OUTLINE.md` §5 for the fuller
discussion-section draft.

---

## Recommended slide count

Start with ~12–14 physical slides, roughly: title/CMV (possibly combined); questions; dataset/
pipeline; length discovery; pivot transition; strategy measurement; strategy-combination gain;
length matching + compact model (may split into two); refutation/justification interpretation; WA
(may split into protocol + results); conclusion/limitations (may split). Prioritize readable charts
and one takeaway per slide. Slide count is secondary to timing.

## Evidence and visual sourcing

- `this work/00_introduction.ipynb`: motivation, CMV context, research questions.
- `this work/01_data_download.ipynb`, `02_data_annotation.ipynb`: collection, dataset, and labels;
  verify exact sampling/negative-construction language in the notebook before final slides.
- `this work/04.1_model_training.ipynb`, `04.2_baseline_comparison.ipynb`: initial model and
  word-count baselines, length distribution.
- `this work/05_rhetorical_scoring.ipynb`: strategy definitions, rubric, scoring, test-retest and
  small human validation.
- `this work/06_strategy_ablation.ipynb`: strategy addition, nine-feature result, order-sensitivity
  checks.
- `this work/04.3_length_controlled_analysis.ipynb`, `06_strategy_ablation.ipynb`: length control.
  Verify denominator and sample used for every displayed retention percentage.
- `this work/07_strategy_interpretation.ipynb`: SHAP and effect sizes, before/after matching.
- `this work/08_wa_external_validation.ipynb`: WA refitting, evaluation, different strategy
  ranking. **Verify exact train/test procedure from source before writing the final methodology
  slide.**

Visuals to prepare: CMV/delta schematic; dataset-to-model pipeline diagram; word-count-by-outcome
histogram; TF-IDF top-token table (optional, supports the length-confound diagnosis); compact
model-comparison bar/dot chart; strategy-gain chart; length-matching retention comparison;
simplified refutation/justification before/after SHAP ranking; WA feature-set comparison table.
None of these exist as standalone exported files yet — reuse notebook plots only if legible on a
projector, otherwise re-render or redraw.

## Methodological wording to preserve

- "Predictive association / predictive information," not "causes persuasion."
- A delta is an observable CMV success label, not a perfect measurement of internal belief change.
- Original dataset model scores and WA scores should not be directly compared without noting
  different class prevalences and sampling.
- In WA, **the model was fitted anew on WA training data** while its selected architecture and
  training configuration were transferred from the original experiments. **Do not call this
  zero-shot transfer or a frozen model.**
- Column-order variation in some XGBoost results means mean±SD over orderings, not a convenient
  single run.
- Strategy scores are LLM-produced; human comparison was small and especially weak for
  refutation/justification.
- Negative-sampling explanation for the corpus-specific ranking reversal is a **hypothesis, not an
  experimentally established cause**.
- Exploratory `wa_quick_experiments.ipynb` work (zero-shot check, embeddings-on-top check,
  re-adding-concession check) is **out of scope** for slides, backup slides, and core report.
  Discuss informally only if asked, with clear qualification.

## Backup slides (held in reserve, time-permitting)

- **A 9-feature model nearly matches/beats the full 237-feature pipeline** (Section 7;
  `06_strategy_ablation.ipynb`) — good if asked "could this be simplified?"
- **The column-order sensitivity finding** (Section 6; `06_strategy_ablation.ipynb`) — good if
  asked "how stable/reproducible are these PR-AUC numbers?" Answer: report as mean±SD across
  orderings, here's why.
- **WA robust-4-feature core vs. the literature's baseline/full-method benchmarks** (Section 9;
  `08_wa_external_validation.ipynb`) — good if asked how this compares to published WA results.

Zero-shot/embeddings-on-top material from `wa_quick_experiments.ipynb` is **excluded from backup
slides entirely** — see "Methodological wording to preserve" above.

## Possible examiner questions, with where the answer lives

| Question | Where the answer is |
|---|---|
| "How do you know the model isn't just learning length again when you say strategies matter?" | Section 7 — the strategy-score *gain* retains 84% of its value under `06`'s own length matching (scored subset); `word_count`'s and the engineered pipeline's own lifts retain only ~18% under `04.3`'s separately-constructed matching (full dataset) — cite these as two different checks, not one chart. |
| "Is refutation really the key strategy, or is that just this one dataset?" | Section 9 — explicitly: no, it reverses on WA; justification dominates there. |
| "How reliable is the LLM's rhetorical scoring?" | Section 8 caveat / `05_rhetorical_scoring.ipynb` — self-consistency strong, human agreement weaker, worst exactly on refutation/justification. |
| "Why PR-AUC and not accuracy or ROC-AUC?" | Section 2 — ~16% positive rate; justified explicitly in `04.1_model_training.ipynb`. |
| "Are these results statistically significant?" | Be honest: most point comparisons have overlapping bootstrap CIs; the word-count and strategy-ablation gains are the two that clear that bar most clearly. |
| "Why didn't you just use a bigger/transformer model end-to-end instead of hand-built features?" | Backup slide — a 9-feature interpretable model nearly matches/beats a 237-feature one, supporting an explicit, defensible interpretability-over-raw-performance choice. |
| "What exactly was refitted on WA? What was held fixed?" | Section 9 — architecture/configuration reused unchanged; a new model instance is fit on WA's own training split. |
| "Why did the dominant strategy change between datasets? What is tested vs hypothesized?" | Section 9 — negative-sampling difference is an explicit, untested hypothesis. |
| "Does this replicate prior published work on CMV?" | Section 1/Related work — contrast with Priniski & Horne (2018, thread-level regression) and Na & DeDeo (2022, unsupervised pattern discovery, whose closest analogue to `refutation` was among their *least* effective categories — a genuine point of tension worth naming rather than avoiding). |

## Decisions / verification before slide production

- [x] Confirm WA refit details directly against `08_wa_external_validation.ipynb` and align all
  prose. *(Verified 2026-10-14 against cell 19: the architecture/configuration selected on the home
  dataset is reused unchanged — no new selection on WA — and a new model is fit on WA's own native
  train split, evaluated on WA's own native heldout split. Confirms this file's existing wording is
  accurate; same correction applied to `REPORT_OUTLINE.md`/`RESULTS_SHORTLIST.md`/
  `READINESS_CHECKLIST.md`.)*
- [ ] Confirm all numbers and their split/sample/metric definitions against notebook outputs.
- [ ] Select which plots can be exported and which need redrawing for readability.
- [ ] Decide if Section 7 needs two physical slides after first timed rehearsal.
- [ ] Time a spoken draft; reduce details before reducing font size.
- [ ] Update `degree_completion/PROGRESS.md` only when work is actually complete.
- [x] **2026-10-14 — resolved.** `06_strategy_ablation.ipynb` cell 36's own markdown claims the
  engineered pipeline's lift "collapsed to 25%" under `04.3`'s strict length matching, citing `04.3`.
  Direct verification against `04.3` cells 19–22 instead finds: (a) the engineered pipeline's own
  strict-match lift retention is ~18.4% (0.0334/0.1817), not 25% — matching `word_count`'s own
  ~17.9%, not differing from it; (b) the only "25%" figure `04.3` actually computes is a *different*
  quantity on a *different* sample — residual `word_count` *correlation* (not lift) remaining after
  the *loose* 84-word floor (`04.3` cell 41: r=0.093/0.3716=25.0%). This is a mix-up inside `06`'s own
  narrative text, not a sample/population-compatibility caveat, and the underlying computed values
  are unaffected. **Resolution: all presentation documentation now states ~18% for both `word_count`
  and the engineered pipeline's strict-match retention, and never cites 25%** — see Section 7 above,
  `RESULTS_SHORTLIST.md` Finding 3, and `figures/FIGURE_NOTES.md`. Initially flagged rather than
  edited, per `RESEARCH_GUARDRAILS.md`'s standing rule; at your explicit request,
  `06_strategy_ablation.ipynb` cell 36's markdown has since also been corrected directly (prose
  only — no code, outputs, or execution counts touched), with an inline note explaining the earlier
  error.
