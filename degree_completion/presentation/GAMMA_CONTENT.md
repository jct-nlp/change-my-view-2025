# Gamma Content — MSc Defense Presentation

**Talk:** Predicting Persuasive Comments on r/ChangeMyView — Length, Rhetoric, and Cross-Corpus
Validation
**Defense date:** 2026-11-04. **Target length:** 14:30 (hard limit 15:00).
**Slides:** 13 main slides + 3 backup slides.
**Figures:** all seven approved figures live in `degree_completion/figures/` as PNG (300 dpi, use
for Gamma upload) and SVG (use if Gamma supports vector upload and crisper scaling is wanted).
Figures are final — do not regenerate, crop, or re-letter them; place them as single images.

This file is Gamma-ready slide content only. Speaker notes are in `SPEAKER_NOTES_HE.md`. Asset
placement detail is in `GAMMA_ASSETS.md`.

---

## Slide 1 — Title & The Delta Mechanism

**Takeaway:** r/ChangeMyView gives a free, per-comment label for persuasive success — the delta —
and this project predicts it.

**Slide text:**
- **Predicting Persuasive Comments on r/ChangeMyView**
  Length, Rhetoric, and Cross-Corpus Validation — MSc Final Project, Dikla Vilge, 2026
- r/ChangeMyView (CMV): a public debate forum. Someone posts a view they're open to changing;
  others reply trying to change it.
- The **delta (Δ)**: if the poster is genuinely persuaded by a specific reply, they award that
  reply a delta — a community-given, per-comment success label.
- **The ML task:** predict which comments earn a delta.

**Layout:** Title slide, large headline, two-line subtitle. Below the fold, a simple two-box
schematic: "OP post" → arrow → "Reply (Δ awarded)" with a small Δ badge icon on the second box.

**Figure:** None approved/exists for this slide. Build fresh — simple schematic, no real post
text (do not fabricate post/comment content). If Gamma auto-generates an illustration, keep it
abstract/iconographic, not a screenshot mockup of an actual Reddit post.

**Source:** `this work/00_introduction.ipynb` §§1.1–1.3.

---

## Slide 2 — Three Research Questions

**Takeaway:** three questions, in increasing specificity, organize the entire talk.

**Slide text:**
- **RQ1.** How well can textual features predict persuasive success, and what role does comment
  length play?
- **RQ2.** Do rhetorical strategies add predictive information beyond length and other textual
  features?
- **RQ3.** Which rhetorical strategies contribute to prediction, and how stable are the findings
  across datasets constructed differently?

**Layout:** Three numbered rows, one RQ per row, generous whitespace. No chart.

**Figure:** None required.

**Source:** `degree_completion/PRESENTATION_STORYLINE.md`, "Three research questions (advisor's
framing)".

---

## Slide 3 — Dataset & Task Definition

**Takeaway:** a concrete, per-comment supervised task, built with a leakage-free evaluation
protocol.

**Slide text:**
- Source: 500 top r/changemyview posts via the Reddit API (PRAW); one row per reply (comment).
- **3,971 comments total — 641 delta-awarded (16.1%), 3,330 not.**
- Negatives are simply the other, non-delta replies already in the same threads — no similarity
  filtering. (Contrast this with the external-validation corpus later.)
- Evaluation: thread-grouped 80/20 split — **zero thread overlap** between train and test
  (verified by assertion).
- Metric: **PR-AUC**, chosen over accuracy/ROC-AUC because only ~16% of comments are positive.
- Model inputs: the engineered-feature model uses reply + original-post + reply↔OP relational
  features; the TF-IDF and word-count baselines use reply text only.

**Layout:** Horizontal 6-step pipeline diagram across the top (data collection → annotation →
feature engineering → modeling → strategy scoring → external validation); a stats callout box
below/beside it with the 3,971 / 641 / 16.1% figures.

**Figure:** None approved/exists. Build fresh — six labeled boxes with arrows; label each box
with its source notebook in small text (01_data_download, 02_data_annotation, 03.x feature
engineering, 04.x modeling, 05–07 strategy scoring, 08 external validation).

**Source:** `00_introduction.ipynb` §1.4; `01_data_download.ipynb`; `02_data_annotation.ipynb`;
`04.1_model_training.ipynb` (split/metric choice). Dataset counts independently verified against
`results/02_comments_df_base.csv.zip`.

---

## Slide 4 — The Length Confound

**Takeaway:** convincing comments are nearly three times longer, on average — the visual root of
the project's central confound.

**Slide text:**
- Mean word count: **78.0** (non-convincing) vs. **216.7** (convincing).
- Correlation: r = 0.3716 (p = 2.92e-130).
- Within-thread check (rules out a pure topic/thread-size artifact): in 200 threads containing
  both classes, the convincing comment outlasted that thread's own non-convincing average in
  **153/200 (76.5%)** (Wilcoxon p = 4.15e-19).
- Only ~16% of comments are convincing at all — why PR-AUC, not accuracy, is used throughout.

**Layout:** Figure large and central (60–65% of slide width); 2–3 bullets to the right or below
with the mean/correlation numbers.

**Figure:** `degree_completion/figures/figure4_comment_length_by_outcome.png`
**Placement instructions:** single full-resolution image, centered, maintain aspect ratio
(~16:9 crop already built in); do not add extra labels on top of the figure — it already carries
its own title/subtitle/footer.

**Source:** `04.2_baseline_comparison.ipynb` cells 30–31; within-thread check from
`RESULTS_SHORTLIST.md` Finding 1.

---

## Slide 5 — The Pivot: Length Beats the Engineered Pipeline

**Takeaway:** a single feature — word count — outperformed a carefully engineered 229-feature
pipeline. This surprising result redirected the entire project.

**Slide text:**
- Full engineered-feature pipeline (229 features, XGBoost + SMOTE, selected via a 27-combination
  grid): **PR-AUC 0.3848**.
- A minimal TF-IDF + Logistic Regression baseline (no engineered features): **PR-AUC 0.5119** —
  beats the full pipeline.
- `word_count` alone (one feature, Logistic Regression): **PR-AUC 0.5256** — the best score yet.
- Diagnosis: TF-IDF's top predictive tokens were mostly stopwords and the token "edit" — a sign
  it is tracking length, not content.
- Fix: folding `word_count` into the engineered features and re-running model selection from
  scratch recovers most of the gap — **PR-AUC 0.4922**, with bootstrap 95% CIs barely overlapping
  the original selection (the single most statistically decisive comparison in the project).
- **"This wasn't a failure — catching it is what let the rest of the project ask a sharper
  question: is there anything beyond length that matters?"**

**Layout:** Figure centered/left; bullets right; framing line set apart in bold at the bottom of
the slide.

**Figure:** `degree_completion/figures/figure1_initial_model_comparison.png`
**Placement instructions:** single image, ~55% slide width; keep the framing-line text outside
the image area so it reads as the presenter's conclusion, not a chart caption.

**Source:** `04.1_model_training.ipynb` (original pipeline numbers); `04.2_baseline_
comparison.ipynb` (TF-IDF baseline, word-count discovery, re-selection with word_count).

---

## Slide 6 — Transition: Is There Anything Beyond Length?

**Takeaway:** predictive association with length is not sufficient to establish that rhetorical
content itself matters — the next test must isolate what survives once length is controlled for.

**Slide text:**
- A model that wins because comments are longer tells us about *length*, not about *what is
  said*.
- **Next question:** do explicitly defined rhetorical strategies add predictive information
  beyond length (RQ2) — and if so, which ones, and how stably (RQ3)?

**Layout:** Minimal, text-only transition slide. Large centered text, no bullets beyond the two
lines above, generous negative space. This slide should visually feel like a breath, not another
data slide.

**Figure:** None — explicit text-only transition per the storyline.

**Source:** `degree_completion/PRESENTATION_STORYLINE.md` Section 4.

---

## Slide 7 — Eight Rhetorical Strategies

**Takeaway:** one compact overview of the measurement instrument — most of the talk's attention
goes to two strategies later, not all eight now.

**Slide text:**
- 8 strategies, each scored **1–5 by an LLM** (Gemini), adapted from Labruna et al. (2026)'s
  persuasion-strategy scoring method: *justification, concession, evidence, refutation,
  personal_narrative, call_to_action, simplification, question_asking.*
- The two that matter most later: **refutation** — directly countering a specific flaw in the
  original argument; **justification** — giving a reason/support beyond bare assertion.
- Scored on comments ≥30 words (or any length if convincing): **2,753 of 3,971 comments.** (Note:
  this subset's convincing rate is 23.3%, not the 16.1% baseline — different denominator.)
- Reliability: strong test-retest self-consistency (95–99% within-1-point agreement across all 8
  strategies); weaker agreement with a single human judge (N=18), weakest specifically on
  refutation and justification. A lightweight check, not a formal reliability study.

**Layout:** A compact row/grid of all 8 strategy labels near the top (equal visual weight, plain
list — do not single out refutation/justification visually here, that comes later); 3 bullets
below for the two-strategy preview, the denominator note, and the reliability caveat.

**Figure:** None essential (per `FIGURE_SELECTION_PLAN.md` Section 5). Build a plain 8-item label
strip if a visual anchor is wanted — not a chart.

**Source:** `05_rhetorical_scoring.ipynb`.

---

## Slide 8 — Strategy Scores Add Real Predictive Signal

**Takeaway:** adding the 8 strategy scores produces a real, order-robust PR-AUC gain beyond the
existing engineered model.

**Slide text:**
- Engineered model alone (scored population, n=2,753): **PR-AUC 0.4467 ± 0.0196**.
- + 8 strategy scores: **PR-AUC 0.5202 ± 0.0177 (+0.0735)** — clears the column-order noise band.
- Strategy scores alone: **PR-AUC 0.3538 ± 0.0079** — real standalone signal, between no-skill
  (0.2382) and the full model.
- Methodological disclosure: this XGBoost+SMOTE pipeline, on 200+ correlated columns, is
  sensitive to column *order* — reshuffling column order alone swings PR-AUC by up to 0.0422
  (traced to XGBoost's tie-breaking on correlated features, not a SMOTE artifact). Every PR-AUC
  number from here on is mean ± SD across multiple column orderings, never a single run.

**Layout:** Figure large/centered; 1–2 bullets below for the disclosure line, set in slightly
smaller text as a methodological footnote the presenter still says out loud.

**Figure:** `degree_completion/figures/figure2_strategy_incremental_value.png`

**Source:** `06_strategy_ablation.ipynb` cell 17 (headline numbers); cells 11–14 (column-order
check).

---

## Slide 9 — Does the Gain Survive Length Control?

**Takeaway:** the strategy-score *gain* survives length matching far better than the feature
families it is layered on top of.

**Slide text:**
- Strategy-score gain: **0.0664** (full scored population, n=2,753) → **0.0557** (length-matched
  sample, n=1,235) — **84% retained**.
- A descriptive point estimate, not a formal significance test — it shows the gain holds up much
  better than other feature families' own lifts under the same kind of matching; it does not
  prove full independence from length.
- Best single combination found anywhere in this project: 8 strategy scores + `word_count` (9
  columns) = **PR-AUC 0.5379 ± 0.0109** — beats every larger combination tested, including the
  full 237-column set. *(Random Forest scores statistically indistinguishably at 0.5432 ± 0.0065
  on this 9-column set — not claiming XGBoost is "the best" architecture here; kept for
  consistency with the rest of the talk.)*

**Layout:** Figure centered/large; two bullet lines below; the 9-feature note in a smaller
font as a one-line aside (full detail reserved for Backup Slide B1 if asked).

**Figure:** `degree_completion/figures/figure3_strategy_gain_length_matching.png`

**Important — do not combine with other numbers:** a separate, differently-built length-matched
sample exists in `04.3_length_controlled_analysis.ipynb` (built from the full 3,971-row dataset,
not the 2,753-row scored population used here). If mentioned at all, state verbally only, as a
separate check: `word_count`'s own lift and the engineered pipeline's own lift each retain only
~18% under that notebook's strict matching — never cite "25%" (a resolved documentation error;
see `RESEARCH_GUARDRAILS.md`). Do not put this on the slide itself.

**Source:** `06_strategy_ablation.ipynb` cells 31–35 (retention), cell 22 (9-feature combination),
cells 28–30 (XGB vs. RF on 9 columns).

---

## Slide 10 — Refutation Holds Up; Justification Doesn't

**Takeaway:** on the home dataset, `refutation` remains important after length control;
`justification`'s apparent importance is mostly a length proxy.

**Slide text:**
- SHAP mean |value|, full population: **refutation 0.6034** (top strategy) » justification
  0.4156 > evidence 0.3592 > concession 0.3439 > simplification 0.3193 > personal_narrative 0.1853
  ≈ call_to_action 0.1837 > question_asking 0.1518.
- SHAP mean |value|, length-matched: **refutation 0.5193** (holds rank, modest decline) >
  concession 0.3445 > simplification 0.3383 > evidence 0.3090 > call_to_action 0.2707 >
  personal_narrative 0.2168 > **justification 0.1956** (collapses from 3rd to 7th-of-8) >
  question_asking 0.1806.
- Cross-check (model-free Mann-Whitney rank-biserial effect size): on the **full population**,
  `justification` (r=0.3584) actually leads `refutation` (r=0.3149) by this measure — the two
  methods disagree on home data's "#1" strategy. Length-matched, only 3 of 8 remain significant:
  refutation (49% retained), justification (37% retained), simplification. `personal_narrative`
  flips to a small, reliable *negative* association.
- **State directly, not as a footnote:** refutation and justification are also the two strategies
  hardest for the LLM to calibrate against human judgment. The strongest signal in this analysis
  comes from the two most noisily-scored strategies.

**Layout:** Figure large/central; bullets beneath in two groups (SHAP vs. effect-size); the
reliability caveat set apart/highlighted, not buried at the bottom.

**Figure:** `degree_completion/figures/figure5_refutation_vs_justification_shap.png`

**Source:** `07_strategy_interpretation.ipynb` cells 36 (SHAP), 39 (effect size); `05_rhetorical_
scoring.ipynb` (reliability caveat).

---

## Slide 11 — External Validation: Protocol and the Reversal

**Takeaway:** the structural finding replicates on an independent corpus (Winning Arguments); the
specific dominant strategy does not.

**Slide text:**
- **Protocol — state precisely:** the architecture/training configuration already selected on
  the home dataset (RobustScaler + SMOTE + XGBoost) is reused unchanged; **no new
  architecture/configuration selection is performed on WA.** A **new model instance is fit on
  WA's own native training split and evaluated on WA's own native heldout split.** This is **not**
  zero-shot transfer and **not** a frozen model applied as-is.
- The reversal: at home, the two importance methods already disagree (`justification` leads by
  effect size, r=0.3584; `refutation` leads by SHAP, 0.603). On WA, `justification` leads by
  **both** SHAP and effect size (r=0.1962); `refutation` drops to 4th by effect size, near-bottom
  by SHAP.
- Not a simple swap: `evidence` makes the single largest jump (6th at home → 2nd on WA);
  `refutation` falls the most among previously-strong strategies (2nd → 4th by effect size).
- **Hypothesis only, explicitly untested:** the two datasets build negative examples differently —
  WA's losing replies are topically Jaccard-matched to the winner; this project's non-convincing
  comments are unfiltered (36.6% under 30 words). One supporting, not proving, number — no
  experiment in this project tests this directly.

**Layout:** Figure centered; the protocol statement boxed/highlighted at the top of the slide
since its exact wording must not be garbled in delivery; reversal bullets below.

**Figure:** `degree_completion/figures/figure6_strategy_ranking_cmv_vs_wa.png`

**Source:** `08_wa_external_validation.ipynb` cell 19 (protocol, verified directly); cell 33 and
`07_strategy_interpretation.ipynb` cell 39 (ranking comparison).

---

## Slide 12 — External Validation: Quantitative Results

**Takeaway:** rhetorical-strategy scores add value on WA too, landing in an honest, bounded
position relative to the published literature — a partial, not full, replication.

**Slide text:**
- WA corpus: 8,106 comments collected (8,096 scored); no-skill PR-AUC baseline **0.5291**;
  word-count confound weaker here (r=0.1516 vs. 0.3716 at home).
- `word_count` only: PR-AUC **0.5767** (lift 0.0476), pairwise accuracy **55.59%**.
- strategy scores only: PR-AUC **0.5967** (lift 0.0676), pairwise accuracy **57.27%**.
- robust 4-feature core (refutation, justification, simplification, `word_count`): PR-AUC
  **0.5919** (lift 0.0628), pairwise accuracy **58.68%**.
- Literature context (Labruna et al., 2026): "Independent Scoring" baselines 53.78–56.66%; full
  MS-PS method 60.59–64.53%. This project's robust core sits above the independent-scoring
  baselines and below the full method — an honest, bounded result, **not** claimed as a
  like-for-like beat of the literature.
- **Limitation — state directly:** this validation supports that strategy scores carry predictive
  information in a second dataset. It does **not** establish that their incremental contribution
  *beyond length* replicates to the same extent as on the home dataset — that would require an
  explicit, comparable length-controlled ablation on WA, which was not done.

**Layout:** Figure (two panels: PR-AUC, pairwise accuracy) centered; bullets below; the limitation
line set apart in italics at the bottom.

**Figure:** `degree_completion/figures/figure7_wa_external_validation.png`

**Source:** `08_wa_external_validation.ipynb` cells 20, 36, 38 (literature numbers), 39.

---

## Slide 13 — Conclusions, Contribution, and Limitations

**Takeaway:** three answers to RQ1–RQ3, in decreasing order of confidence, framed as predictive —
not causal.

**Slide text:**
- **RQ1.** Comment length is the single dominant predictive signal in this dataset — true,
  consistently, everywhere it was tested.
- **RQ2.** A real, length-independent rhetorical signal exists beyond length — also true, and it
  replicates structurally on an independent corpus.
- **RQ3.** That signal concentrates specifically in `refutation` on this project's own data — true
  here, but it did not replicate in that specific form on WA, where `justification` took its
  place. Only an explicitly labeled, untested hypothesis exists for why.
- **Limitations (state directly):** single subreddit, single platform, English only. LLM-scored
  strategy reliability is weakest on exactly the two strategies that matter most for the headline
  claims. Most individual point-estimate comparisons in this project have overlapping confidence
  intervals at this sample size — the comparisons strong enough to call decisive are the
  word-count discovery (Slide 5) and the strategy-ablation gain (Slide 8). A delta is an
  observable CMV success label, not a complete or error-free measurement of internal belief
  change.
- **Future work (explicitly future, not completed):** directly test the negative-sampling
  hypothesis from Slide 11; extend validation to a third, differently-constructed corpus.

**Layout:** A condensed 3-row RQ recap at the top (short, bolded one-liners); limitations as a
bullet list below; one closing line for future work.

**Figure:** None required — optionally a plain 3-row RQ recap table reusing wording already
established earlier in the talk; no new chart.

**Source:** synthesis across all notebooks; `degree_completion/RESEARCH_GUARDRAILS.md`;
`REPORT_OUTLINE.md` §5.

---

## Backup Slides (held in reserve, time-permitting or by request)

### Backup B1 — Could a 9-Feature Model Replace the Full Pipeline?

**Use if asked:** "Could this be simplified?" / "Why not just throw in everything?"

**Slide text:**
- 9-column combination (8 strategy scores + `word_count`): **PR-AUC 0.5379 ± 0.0109** vs. the
  full 237-column set's **0.5202 ± 0.0177** — a small, interpretable model beating a much larger
  one, not a close second.
- Architecture note: re-running the full classifier×resampler grid on just these 9 columns found
  Random Forest edges out XGBoost (0.5432 ± 0.0065 vs. 0.5380 ± 0.0108) — a gap smaller than
  either architecture's own order-induced spread, so **not** a confident win either way. XGBoost
  is kept for consistency with the rest of the talk, not because it was shown to outperform
  Random Forest on this feature set.

**Layout:** Simple text/table slide — two rows, two numbers each, no chart needed.

**Figure:** None approved for this specific comparison; present as a plain text/table slide.

**Source:** `06_strategy_ablation.ipynb` cells 22–23 (9-column combination), cells 28–30
(XGB vs. RF on 9 columns).

### Backup B2 — How Stable Are These PR-AUC Numbers? (Column-Order Sensitivity)

**Use if asked:** "How reproducible are these results?" / "What do the error bars mean?"

**Slide text:**
- On this correlated 200+-column feature matrix, reshuffling only column *order* (identical data,
  identical architecture) swings PR-AUC by as much as **0.0422**.
- Root cause: XGBoost's deterministic tie-breaking among near-equal split gains on correlated
  features — **not** a SMOTE artifact.
- Every PR-AUC number from Slide 8 onward is reported as mean ± SD across multiple column
  orderings for exactly this reason — never a single convenient run.

**Layout:** Text-only slide, 2–3 short bullets.

**Figure:** None approved; text-only.

**Source:** `06_strategy_ablation.ipynb` cells 11–14.

### Backup B3 — Could WA's Result Be Pushed Further?

**Use if asked:** "Could you do even better on WA?" / deeper benchmark comparison.

**Slide text:**
- The single best configuration found by pruning the strategy-feature set on WA
  (removing `call_to_action`, 8 of 9 columns): **PR-AUC 0.5937**, pairwise accuracy **59.10%** —
  slightly better than the robust 4-feature core on both metrics.
- **Not shown on the main results slide deliberately:** the notebook's own framing treats the
  pre-registered 4-feature robust core, not a variant scanned for the best score, as the
  principled number to report. Showing the best-found number as the headline result would read as
  tuning on WA — which this project's protocol explicitly avoids.

**Layout:** Text-only slide, reuse Figure 7 as a visual backdrop/reference if desired (do not
alter the figure itself).

**Figure:** `degree_completion/figures/figure7_wa_external_validation.png` optionally reused as
backdrop; no new figure.

**Source:** `08_wa_external_validation.ipynb` cell 37 (markdown framing) and the pruning table
referenced there.

---

## Global notes for building this in Gamma

- One main claim per slide; every plot/table should state dataset, metric, split/sample, and
  whether numbers are mean±SD, either on the image itself (already true for all seven approved
  figures) or in the slide bullets.
- Never present correlations or SHAP/effect-size importances as causal effects — this wording is
  already built into the bullets above; preserve it if Gamma's AI rewrites or shortens any slide.
- Never call the WA procedure "zero-shot" or "the same model applied to WA" — Slides 11–12 and
  Backup B3 state the refit protocol explicitly; keep that wording intact.
- Keep the two differently-built "length-matched sample" results (Slide 9's 84% vs. the verbal-
  only ~18% aside from `04.3`) visually and textually separate — never merge into one chart,
  table, or bullet.
- Do not reference `wa_quick_experiments.ipynb` (the excluded zero-shot check) anywhere, including
  backup slides.
