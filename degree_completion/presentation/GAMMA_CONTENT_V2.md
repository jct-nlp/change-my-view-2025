# Gamma Content V2 — Visual-First Revision

**Talk:** Predicting Persuasive Comments on r/ChangeMyView — Length, Rhetoric, and Cross-Corpus
Validation
**Defense date:** 2026-11-04. **Target length:** 14:30 (hard limit 15:00).
**Slides:** 13 main slides + 3 backup slides (same narrative sequence as V1).

**What changed from V1, and why:**
- V1's slides carried full bullet lists of methodology, numbers, and caveats. Every main slide
  here is cut to a headline + (where one exists) one approved figure + at most three very short
  lines of on-slide text. Everything else V1 had as slide text — numbers, qualifications,
  transitions — moved into `SPEAKER_NOTES_HE_V2.md`, matched to the slide it came from.
- All seven approved figures are now explicitly called out as the **primary visual** of their
  slide (V1 named the files but the actual Gamma export dropped them) — see
  `GAMMA_ASSETS_V2.md` for exact filenames and upload instructions.
- Two phrasings corrected per guardrails: V1's conclusion slide called the strategy-score signal
  "length-independent" — this overstates what length matching actually shows (a gain that
  *survives* matching better than other signal, not proven independence). Fixed on Slide 13.
  V1's WA slide said the finding "replicates" without qualification in one spot — this is a
  **partial** replication (the group-level signal recurs; the specific dominant strategy does
  not) and is now stated as such directly on Slide 12's headline, not only in notes.
- Figures are used exactly as approved — no new figure, diagram content, or statistic was
  invented for this revision; all on-slide numbers already appeared in V1 and trace to the same
  notebook cells.

**Figures:** all seven live in `degree_completion/figures/` as PNG (300 dpi — use for Gamma
upload) and SVG. Do not regenerate, crop, or re-letter them.

This file is Gamma-ready slide content only. Hebrew speaker notes are in
`SPEAKER_NOTES_HE_V2.md`. Figure filenames and placement are in `GAMMA_ASSETS_V2.md`.

---

## Slide 1 — Title & The Delta Mechanism

- **Headline:** What Makes a Reply Persuasive?
- **Primary visual:** Fresh two-box schematic — "OP post" → arrow → "Reply (Δ awarded)" with a
  small delta badge icon on the second box. No approved figure exists for this; build directly in
  Gamma, abstract/iconographic only — do not fabricate real post or comment text.
- **Minimal on-slide text:**
  - r/ChangeMyView: argue well enough to earn a **Δ (delta)**
  - Goal: predict which replies earn one
- **What is spoken but not shown:** full explanation of CMV's culture and rules (the OP must be
  genuinely open to changing their mind); that the delta is the community's own long-standing
  mechanism, not something invented for this project; that it gives a free, per-*comment* label
  rather than a vague thread-level outcome; one sentence of motivation (moderation, persuasive-
  writing assistance) without dwelling on it; presenter/date framing.

**Source:** `this work/00_introduction.ipynb` §§1.1–1.3.

---

## Slide 2 — Three Research Questions

- **Headline:** Three Questions, Increasingly Specific
- **Primary visual:** A simple vertical funnel/arrow diagram — three stacked boxes narrowing left
  to right or top to bottom, labeled briefly (see on-slide text). Build fresh in Gamma; no
  approved figure exists for this.
- **Minimal on-slide text:**
  - RQ1 — Does length predict persuasion?
  - RQ2 — Is there signal beyond length?
  - RQ3 — Which strategies, and how stable?
- **What is spoken but not shown:** the full RQ wording ("How well can textual features predict
  persuasive success, and what role does comment length play?" etc.); that the whole talk follows
  this exact progression — prediction, length confound, testing beyond length, identifying a
  specific strategy, external validation and its limits.

**Source:** `degree_completion/PRESENTATION_STORYLINE.md`, "Three research questions."

---

## Slide 3 — Dataset & Task Definition

- **Headline:** 3,971 Comments, One Task
- **Primary visual:** Horizontal 6-step pipeline diagram — data collection → annotation → feature
  engineering → modeling → strategy scoring → external validation. Build fresh in Gamma (no
  approved figure exists); keep each box label short, small sub-label with source notebook number
  if space allows.
- **Minimal on-slide text:**
  - 641 persuasive / 3,971 total (16.1%)
  - Thread-disjoint train/test split
  - Metric: PR-AUC (rare positive class)
- **What is spoken but not shown:** 500 posts collected via Reddit's API (PRAW); one row per
  reply; negatives are simply other same-thread replies with no similarity filtering (contrast
  with WA later); zero thread overlap verified by assertion, to prevent leakage; the
  engineered-feature model uses reply + original-post + reply↔OP relational features, while
  the TF-IDF/word-count baselines use reply text only.

**Source:** `00_introduction.ipynb` §1.4; `01_data_download.ipynb`; `02_data_annotation.ipynb`;
`04.1_model_training.ipynb` (split/metric choice). Counts verified against
`results/02_comments_df_base.csv.zip`.

---

## Slide 4 — The Length Confound

- **Headline:** Convincing Comments Are Much Longer
- **Primary visual:** `figure4_comment_length_by_outcome.png` — full width/dominant, ~65–70% of
  slide. The chart's own title, subtitle, and footer already carry the key numbers (78.0 vs.
  216.7 words); do not duplicate them as separate slide text.
- **Minimal on-slide text:** none beyond the headline — let the chart carry the comparison.
- **What is spoken but not shown:** exact correlation (r = 0.3716, p = 2.92e-130); the
  within-thread check that rules out a pure topic/thread-size artifact (153/200 threads, 76.5%,
  Wilcoxon p = 4.15e-19); that only ~16% of comments are convincing at all, which is why PR-AUC
  rather than accuracy is used throughout the talk.

**Source:** `04.2_baseline_comparison.ipynb` cells 30–31; within-thread check from
`RESULTS_SHORTLIST.md` Finding 1.

---

## Slide 5 — The Pivot: Length Beats the Engineered Pipeline

- **Headline:** A Single Number Beat 229 Features
- **Primary visual:** `figure1_initial_model_comparison.png` — large/central, ~60% slide width.
  Chart already shows all three PR-AUC values; no need to restate them as bullets.
- **Minimal on-slide text:**
  - word count alone beat the full engineered pipeline
- **What is spoken but not shown:** the three exact numbers (0.3848 engineered pipeline, 0.5119
  TF-IDF baseline, 0.5256 word-count-alone); the diagnosis (TF-IDF's top tokens were mostly
  stopwords plus "edit" — a sign of tracking length, not content); the fix (folding word_count
  into the engineered features recovers most of the gap, to 0.4922, with bootstrap 95% CIs barely
  overlapping — the single most statistically decisive comparison in the project); the framing
  line that this was not a failure but what sharpened the rest of the project's question.

**Source:** `04.1_model_training.ipynb` (original pipeline numbers); `04.2_baseline_
comparison.ipynb` (TF-IDF baseline, word-count discovery, re-selection with word_count).

---

## Slide 6 — Transition: Beyond Length?

- **Headline:** Length Isn't the Whole Story
- **Primary visual:** None — deliberately a text-only breathing point between the talk's two
  halves. If a visual anchor is wanted, a single large minimal icon (e.g., an arrow or a plain
  question mark) is enough; do not manufacture a chart for a 30-second pivot line.
- **Minimal on-slide text:**
  - Is there a rhetorical signal beyond length?
- **What is spoken but not shown:** that a model winning because comments are longer tells us
  about length, not about what is said; the explicit statement of RQ2/RQ3 as the next test.

**Source:** `degree_completion/PRESENTATION_STORYLINE.md` Section 4.

---

## Slide 7 — Eight Rhetorical Strategies

- **Headline:** Eight Strategies, Scored by LLM
- **Primary visual:** A plain horizontal row of 8 equal-weight labels (justification, concession,
  evidence, refutation, personal_narrative, call_to_action, simplification, question_asking) —
  build fresh in Gamma as a simple label strip, not a chart. Keep all eight visually equal here;
  do not highlight refutation/justification yet — that distinction belongs to Slides 10–11, not
  this overview.
- **Minimal on-slide text:**
  - Scored 1–5 by LLM (Gemini) on 2,753 comments
  - Focus later: refutation & justification
- **What is spoken but not shown:** one-line definitions of refutation (countering a specific
  flaw) and justification (giving a reason beyond bare assertion); the scoring-population
  denominator shift (23.3% convincing in this subset vs. 16.1% baseline — a different base, not a
  contradiction); reliability numbers (95–99% test-retest self-consistency across all 8
  strategies; weaker single-human-judge agreement, weakest specifically on refutation and
  justification; explicitly a lightweight check, not a formal reliability study).

**Source:** `05_rhetorical_scoring.ipynb`.

---

## Slide 8 — Strategies Add Real Predictive Signal

- **Headline:** Strategies Add Real Predictive Signal
- **Primary visual:** `figure2_strategy_incremental_value.png` — large/centered, ~65% width. The
  +0.0735 gain bracket is already drawn on the chart.
- **Minimal on-slide text:**
  - Order-robust gain (mean ± SD across column orderings)
- **What is spoken but not shown:** the three exact PR-AUC values (0.4467±0.0196 engineered
  alone, 0.5202±0.0177 with strategies, 0.3538±0.0079 strategies alone); the column-order
  sensitivity finding itself (reshuffling column order alone swings PR-AUC by up to 0.0482,
  root-caused to XGBoost's deterministic tie-breaking on correlated features, not a SMOTE
  artifact); that every PR-AUC number from this point in the talk onward is reported the same
  way, for exactly this reason.

**Source:** `06_strategy_ablation.ipynb` cell 17 (headline numbers); cells 11–14 (column-order
check).

---

## Slide 9 — Does the Gain Survive Length Control?

- **Headline:** The Gain Survives Length Matching
- **Primary visual:** `figure3_strategy_gain_length_matching.png` — centered, ~60% width. The
  "84% retained" callout and the descriptive-estimate caveat are already on the chart.
- **Minimal on-slide text:**
  - Still associated with length — not proof of full independence
- **What is spoken but not shown:** the two exact numbers (0.0664 full population → 0.0557
  length-matched, n=2,753 → n=1,235); that this is a decile-bin match, not exact pairing; that
  this is a descriptive point estimate, not a formal significance test; the separate, verbal-only
  aside that a *different* notebook, on the *full* 3,971-row dataset, found word_count's own lift
  and the engineered pipeline's own lift each retain only ~18% under a differently-built length
  match — two different checks, never merged into one chart or claim; the 9-feature combination
  aside (0.5379±0.0109, best found anywhere in the project) as a one-line pointer to Backup B1 if
  time allows.

**Source:** `06_strategy_ablation.ipynb` cells 31–35 (retention), cell 22 (9-feature
combination).

---

## Slide 10 — Refutation Holds Up; Justification Fades

- **Headline:** Refutation Holds Up; Justification Fades
- **Primary visual:** `figure5_refutation_vs_justification_shap.png` — the densest figure in the
  deck; give it the most space of any slide, ~70% width.
- **Minimal on-slide text:**
  - By SHAP, after length control
  - Also the 2 least human-reliable scores
- **What is spoken but not shown:** the full SHAP values, full population and length-matched, for
  all 8 strategies; that a *different*, model-free method (Mann-Whitney rank-biserial effect
  size) actually ranks `justification` first at home (r=0.3584, ahead of refutation's 0.3149) —
  the two importance methods disagree on home data's "#1" strategy, and that disagreement must be
  stated, not hidden behind the SHAP chart alone; that under length matching only 3 of 8
  strategies remain significant by that second method; that refutation and justification are also
  the two strategies the LLM scorer agreed with a human judge on the least — the strongest signal
  in this analysis comes from the two most noisily-measured strategies.

**Source:** `07_strategy_interpretation.ipynb` cells 36 (SHAP), 39 (effect size); `05_rhetorical_
scoring.ipynb` (reliability caveat).

---

## Slide 11 — Same Method, New Corpus — Different Top Strategy

- **Headline:** Same Method, New Corpus — Different Top Strategy
- **Primary visual:** `figure6_strategy_ranking_cmv_vs_wa.png` — centered, ~60% width.
- **Minimal on-slide text:**
  - Refit on WA's own data — not zero-shot
  - Justification leads on WA; refutation does not
- **What is spoken but not shown:** the precise protocol (architecture/configuration selected on
  the home dataset reused unchanged; a *new* model instance fit on WA's own training split,
  evaluated on WA's own heldout split — explicitly not a frozen model applied as-is); that this
  is not a simple two-strategy swap — `evidence` makes the single largest jump (6th at home → 2nd
  on WA) and `refutation` falls the most among previously-strong strategies; the author's own
  negative-sampling explanation, stated explicitly as an **untested hypothesis**, not a
  demonstrated cause, with its one supporting (not proving) number.

**Source:** `08_wa_external_validation.ipynb` cell 19 (protocol, verified directly); cell 33 and
`07_strategy_interpretation.ipynb` cell 39 (ranking comparison).

---

## Slide 12 — Partial Replication, Honestly Bounded

- **Headline:** Partial Replication, Honestly Bounded
- **Primary visual:** `figure7_wa_external_validation.png` — two-panel image, centered, ~70%
  width (wider than other slides since it has two panels).
- **Minimal on-slide text:**
  - Strategy signal holds; the dominant strategy does not
  - 58.7% pairwise accuracy — between literature baselines
- **What is spoken but not shown:** the exact feature-set comparison (word_count only PR-AUC
  0.5767 / 55.59% pairwise; strategies only 0.5967 / 57.27%; robust 4-feature core 0.5919 /
  58.68%); the literature reference bands (Labruna et al. 2026: independent-scoring baselines
  53.78–56.66%, full method 60.59–64.53%) as context only, not a like-for-like benchmark; the
  explicit limitation that this validates the strategy signal existing in a second dataset, but
  does **not** establish that its incremental contribution *beyond length* replicates to the same
  extent as at home — that would need a comparable length-controlled ablation on WA, not done
  here.

**Source:** `08_wa_external_validation.ipynb` cells 20, 36, 38 (literature numbers), 39.

---

## Slide 13 — Three Answers, Decreasing Confidence

- **Headline:** Three Answers, Decreasing Confidence
- **Primary visual:** A simple 3-row qualitative recap — one row per RQ, each with a short status
  tag (e.g., a checkmark for "confirmed," a half-check or arrow icon for "partial"). Build fresh
  in Gamma; this restates conclusions already established earlier in the talk, not a new
  statistic or chart — do not attach invented numbers or confidence percentages to the rows.
- **Minimal on-slide text:**
  - RQ1 — Length dominates: confirmed
  - RQ2 — Signal beyond length: confirmed, not proof of independence
  - RQ3 — Which strategy: partial replication only
- **What is spoken but not shown:** the full limitations list (single subreddit/platform/
  language; LLM-scoring reliability weakest on exactly the two strategies the headline claims
  depend on; most point-estimate comparisons have overlapping confidence intervals at this
  sample size — the decisive ones are the word-count discovery and the strategy-ablation gain;
  a delta is an observable CMV success label, not a complete or error-free measurement of
  internal belief change); future work stated explicitly as future (testing the negative-sampling
  hypothesis directly; a third, differently-constructed corpus).

**Correction from V1:** do not describe the RQ2 signal as "length-independent" on this slide or
in speech — length matching shows the gain *survives* better than other signal, not that it is
proven independent of length. Do not describe the WA result as replication without the word
"partial" — the group-level signal recurs; the specific dominant strategy does not.

**Source:** synthesis across all notebooks; `degree_completion/RESEARCH_GUARDRAILS.md`.

---

## Backup Slides (held in reserve, time-permitting or by request)

Backup slides may carry slightly more on-slide text than main slides since they are reserve/Q&A
material, not part of the timed narrative — but still prefer a compact table over paragraphs.

### Backup B1 — Could a 9-Feature Model Replace the Full Pipeline?

- **Headline:** Nine Features Beat Two Hundred Thirty-Seven
- **Primary visual:** None approved; a simple two-row table (9-feature vs. 237-feature PR-AUC).
- **Minimal on-slide text:** 9 features: 0.5379±0.0109 · 237 features: 0.5202±0.0177
- **What is spoken but not shown:** that Random Forest edges out XGBoost on this 9-column set
  (0.5432±0.0065 vs. 0.5380±0.0108) — a gap smaller than either architecture's own order-induced
  spread, so not a confident win either way; XGBoost is kept for consistency with the rest of the
  talk, not because it beat Random Forest on this feature set.

**Source:** `06_strategy_ablation.ipynb` cells 22–23, 28–30.

### Backup B2 — How Stable Are These Numbers?

- **Headline:** Column Order Alone Moves the Score
- **Primary visual:** None approved; text-only.
- **Minimal on-slide text:** Reordering columns alone swings PR-AUC by up to 0.0482
- **What is spoken but not shown:** the root cause (XGBoost's deterministic tie-breaking among
  near-equal split gains on correlated features, not a SMOTE artifact); that this is why every
  PR-AUC number from Slide 8 onward is mean±SD across orderings, never a single convenient run.

**Source:** `06_strategy_ablation.ipynb` cells 11–14.

### Backup B3 — Could WA's Result Be Pushed Further?

- **Headline:** A Slightly Better WA Configuration Exists — and Was Set Aside
- **Primary visual:** `figure7_wa_external_validation.png` may be reused as a visual backdrop
  (same file as Slide 12, not a new asset).
- **Minimal on-slide text:** Best found (−call_to_action): 0.5937 PR-AUC / 59.10% pairwise
- **What is spoken but not shown:** that this was deliberately not shown as the headline WA
  result, because the notebook's own framing treats the pre-registered 4-feature robust core, not
  a variant scanned for the best score, as the principled number — showing the best-found number
  as the main result would read as tuning on WA, which this project's protocol explicitly avoids.

**Source:** `08_wa_external_validation.ipynb` cell 37 (markdown framing) and its pruning table.

---

## Global notes for building this in Gamma

- One headline, one visual, at most three short lines per main slide — if Gamma's AI rewriting
  expands a slide back toward paragraphs, trim it back to this shape rather than accepting the
  expansion.
- Never let Gamma auto-generate a substitute chart for Slides 4, 5, 8, 9, 10, 11, or 12 — these
  must use the uploaded approved figure, not an AI-generated chart from the bullet numbers.
- Never state the RQ2 rhetorical signal as "independent of length," and never state the WA result
  as replication without "partial" — see Slide 13's correction note; this applies to every slide
  that touches either claim (9, 11, 12, 13).
- Never call the WA procedure "zero-shot" or "the same model applied to WA" (Slides 11–12,
  Backup B3).
- Keep the two differently-built "length-matched sample" results (Slide 9's 84% vs. the
  verbal-only ~18% aside from a different notebook) textually separate — never merge into one
  number, chart, or bullet, on-slide or in speech.
- Do not reference `wa_quick_experiments.ipynb` anywhere, including backup slides.
