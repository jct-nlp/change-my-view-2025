# Report Outline — M.Sc. Final Project

Scope note: this outline covers the university-required report structure, mapped onto the existing
research in `this work/`. It does not write the report. Every item below is marked:

- **[VERIFIED]** — confirmed directly from notebook code/output/markdown, with exact source.
- **[INTERPRETIVE]** — a reasonable narrative framing or connection between verified facts, not
  itself a number pulled from a notebook. Safe to use as prose, but should read as interpretation.
- **[MISSING]** — not yet available in any notebook/file; needs to be produced before the report
  section can be completed.
- **[NEEDS VERIFICATION]** — plausible but not independently confirmed by reading the notebook's
  own output; check before citing.

Dataset numbers used throughout this outline were directly verified by reading
`results/02_comments_df_base.csv.zip`: **3,971 comments, 641 convincing (16.1%), 3,330 not
convincing (83.9%)**. Where a notebook reports a different percentage against the same underlying
641 convincing comments (e.g. 23.3%), it is because the denominator shrank — see the 05/06/07 note
under Methods §3.6 below. Always state which denominator a percentage uses.

---

## 1. Abstract (≤250 words)

**Cover:** one paragraph each on: the problem (predicting persuasive, delta-awarded comments on
r/changemyview), the method (engineered features + an 8-strategy LLM-scored rhetorical rubric,
evaluated with a thread-grouped honest split), the two headline findings (comment length confounds
naive comparisons, but a length-independent rhetorical signal survives and concentrates in
`refutation`), and the external-validation result (the structural finding generalizes to an
independent corpus, but which specific strategy dominates does not).

**Source:** write last, after Results/Discussion are drafted — an abstract summarizing sections
that don't exist yet will drift from the final content.

**Status: [MISSING]** — not written. Not a verification gap, a drafting-order one.

---

## 2. Introduction

### 2.1 Motivation
**Cover:** why persuasion on CMV is worth studying; practical applications (content moderation,
debate analysis, persuasive-writing assistance).
**Source:** `this work/00_introduction.ipynb` §1.3 (verbatim motivating paragraph already exists,
can be adapted directly) — **[VERIFIED]**.

### 2.2 Problem definition
**Cover:** state the four research sub-goals exactly as scoped in the project (textual
differentiators of persuasion; the role of comment length; a predictive classifier; whether a
defined rhetorical-strategy rubric adds signal beyond simpler features and length; external
validation on an independent corpus).
**Source:** `00_introduction.ipynb` §1.2, verbatim goal list — **[VERIFIED]**.

### 2.3 Related work
**Cover three works, each with a concrete methodological or empirical contrast to this project:**

1. **Priniski & Horne (2018)**, *"Attitude Change on Reddit's Change My View"*, Proc. 40th Annual
   Meeting of the Cognitive Science Society (ASU). **[VERIFIED]** — PDF at
   `previous work/study 1/cogsci-reddit.pdf`. This is the project's direct methodological ancestor:
   `01_data_download.ipynb`/`02_data_annotation.ipynb` are explicitly adapted from these authors'
   own `collect_posts.py`/`discussion_analysis.py` (github.com/jpriniski/CMV). Their study: 500 top
   CMV posts, >100,000 comments, thread/topic-level Bayesian regression; sociomoral threads get more
   participation/evidence but similar delta rates to non-sociomoral threads; evidence citation
   predicts deltas regardless of topic. Contrast to draw explicitly: they study **thread-level**
   factors with regression; this project builds a **per-comment** supervised classifier with
   engineered features, embeddings, and an LLM-scored rhetorical rubric, plus external-corpus
   validation.
2. **Na & DeDeo (2022)**, *"The Diversity of Argument-Making in the Wild... Reddit's Change My
   View"*, arXiv:2205.07938. **[VERIFIED]** — PDF at `previous work/diversity of arguments/2205.07938v1.pdf`.
   Unsupervised discovery (PMI network + Louvain clustering) of 6 argument-pragmatics patterns
   across 100,170 posts / 5.8M replies; "Causation & Examples" most effective (+23.0% delta-bonus),
   "Relevance & Presumption" least (−19.4%). Contrast to draw: bottom-up/unsupervised discovery vs.
   this project's top-down, fixed 8-strategy rubric scored by an LLM. **Worth one explicit sentence
   in Discussion** — Na & DeDeo's "Relevance & Presumption" category is the closest prior-work
   analogue to this project's `refutation` strategy, and they found it among the *least* effective,
   while this project finds `refutation` to be its *most* robust predictor — a genuine point of
   tension with prior work worth naming, not glossing over. **[INTERPRETIVE]** framing of the
   tension; the two numbers themselves are each **[VERIFIED]** from their respective sources.
3. **Labruna, Modzelewski, Satta & Da San Martino (2026)**, *"Detecting Winning Arguments with
   Large Language Models and Persuasion Strategies"*, Findings of the Association for
   Computational Linguistics: EACL 2026, pages 1888–1915, Rabat, Morocco. **[VERIFIED
   2026-10-14]** — confirmed directly against the published paper (ACL Anthology
   2026.findings-eacl.97): title, author order, and venue all match `08`'s citation exactly; page
   numbers added (not previously known). This is the paper whose method this project adapts
   (LLM-assigned persuasion-strategy scores) — their own taxonomy of 6 strategies (Attack on
   Reputation, Distraction, Manipulative Wording, Simplification, Justification, Call) is drawn
   from prior propaganda/persuasion-technique literature (Dimitrov et al. 2024; Piskorski et al.
   2023a,b) and applied by Labruna et al. primarily to Winning Arguments, Anthropic/Persuasion,
   and Persuasion for Good — a SemEval 2023 news-article dataset (536 articles) is used only as a
   secondary human-annotation validation check (§5.5), not the paper's primary application domain,
   so "6 propaganda-technique strategies on news articles" slightly overstates how central news
   text is to their main experiments; minor wording to tighten if this sentence is reused in the
   final report. This project adapts the *method* to 8 new, CMV-specific constructive strategies
   (refutation, justification, concession, evidence, personal narrative, call-to-action,
   simplification, question-asking) — two names (`justification`, `simplification`) are shared with
   Labruna et al.'s own taxonomy but defined independently for this project's CMV rubric, not
   reused from their definitions. Also the source of the Winning Arguments pairwise-accuracy
   benchmark numbers compared against in §08: **confirmed exactly against the paper's own Table 1**
   — Independent Scoring 53.78–56.66% (min Gemma-3-12B, max LLaMA-3.1-8B), MS-PS-AVG 60.59–62.83%
   (min OpenAI-o3, max Gemma-3-12B), best reported 64.53% (MS-PS-MLP, OpenAI-o3). All three ranges
   match `08`'s citation exactly, with no discrepancy found.
4. **Tan, Niculae, Danescu-Niculescu-Mizil & Lee (2016)**, *"Winning Arguments: Interaction
   Dynamics and Persuasion Strategies in Good-faith Online Discussions"*, Proceedings of the 25th
   International Conference on World Wide Web (WWW 2016), pages 613–624. **[VERIFIED 2026-10-14]**
   — confirmed directly: title, author order, and venue all match `08`'s citation exactly; page
   numbers added (not previously known). Source of the external-validation corpus used in §08; the
   dataset's own pairing methodology (4,263 winner/loser pairs, matched by Jaccard similarity of
   initial replies within the same thread) is independently corroborated by Labruna et al.'s
   description of the same corpus (§3.1 of their paper) and matches `08_wa_external_validation.ipynb`'s
   own pair-reconstruction assertion (`assert len(pairs_df) == 4263`).

### 2.4 What is different in this project
**Cover, as a short explicit list — already-verified contrasts:**
- Per-comment (not thread/corpus-level) supervised prediction with a thread-grouped, leakage-free
  split — **[VERIFIED]**, `04.1_model_training.ipynb`.
- A fixed, 8-strategy, LLM-scored rhetorical rubric purpose-built for CMV, validated against both
  self-consistency and (limited) human hand-scoring — **[VERIFIED]**, `05_rhetorical_scoring.ipynb`.
- An explicit methodological pivot after discovering comment length confounds the naive model
  comparison, with a dedicated length-control analysis isolating what survives — **[VERIFIED]**,
  `04.2`/`04.3`/`06`/`07`. This self-correction (catching and fixing an initial result rather than
  reporting it uncritically) is itself a point worth narrating in the report, not just the final
  numbers — **[INTERPRETIVE]** framing choice.
- External validation on an independent corpus built with a different sampling methodology
  (topically-matched pairs rather than this project's own thread-leaf sampling), with an honestly
  reported partial (not full) replication — **[VERIFIED]**, `08_wa_external_validation.ipynb`.

### 2.5 Learning-system architecture
**Cover:** an end-to-end pipeline diagram/description — data collection → annotation → feature
engineering (3 notebook stages) → initial model selection (honest split, PR-AUC) → baseline
investigation (confound discovery) → length control → rhetorical scoring → strategy
ablation/interpretation → external validation. **[VERIFIED]**, directly mirrors `00_introduction.ipynb`
§1.4's own 5-stage process description, which can be extended to the fuller notebook sequence per
the README table (`this work/00` through `08`).
**Figure to produce:** a pipeline diagram (notebooks as stages, arrows for data flow, each stage's
output file named) — **[MISSING]**, does not exist yet as a figure; should be drawn fresh for the
report (not extracted from a notebook, since no notebook contains this diagram).

### 2.6 Selected learning methods
**Cover, each with one sentence why it was chosen (expand in Methods, just name here):**
XGBoost (final classifier, chosen after a validation grid over 3 classifiers × 3 resampling
strategies), SMOTE (minority oversampling for a ~16% positive rate), TF-IDF+Logistic Regression
(interpretable baseline that turned out to beat the engineered model), Gemini LLM scoring (for
sentiment/tone/style/rhetorical-appeal features where lexicon/classifier methods failed
qualitatively), document embeddings (BAAI/bge-base-en-v1.5), SHAP (model interpretation),
Mann-Whitney U rank-biserial effect sizes (model-free interpretation, cross-checked against SHAP).
**[VERIFIED]**, across `04.1`, `04.2`, `03.1`–`03.2`, `07`.

### 2.7 Relevance and limitations
**Cover:** practical relevance already stated in 2.1; limitations to flag here at a high level
(detailed versions belong in Discussion): single-subreddit dataset, English-only, LLM-scored
features carry their own reliability ceiling (see §3.6 below). **[VERIFIED]** facts,
**[INTERPRETIVE]** framing of which to foreground. (`refutation_reannotation/` is out of scope for
this round of work per explicit user instruction — not a limitation to mention here; see
`READINESS_CHECKLIST.md`.)

---

## 3. Methods

### 3.1 Data collection and annotation
**Cover:** PRAW-based collection of the top 500 r/changemyview posts (`01_data_download.ipynb`,
adapted from Priniski & Horne's own collection code); parsing into one row per comment with
`original_post`, `thread_text` (same-author prior turns only — an explicit, justified design
choice, see below), `final_comment`, `is_convincing` (delta-awarded) labels
(`02_data_annotation.ipynb`).
**Exact numbers [VERIFIED]:** 425 posts loaded → 3,974 raw comment rows → 3 rows removed in a
documented data-quality cleanup (a delta-awarded comment edited post-hoc into placeholder junk
text; an automated moderator bot notice mistaken for a real comment; a comment whose body is a
grid of numbers that broke a readability feature) → **final 3,971 comments, 641 convincing (16.1%),
3,330 not convincing**, independently confirmed by reading `results/02_comments_df_base.csv.zip`
directly in this review.
**Methods-worthy design choice to quote [VERIFIED]:** `thread_text` is defined as only the same
commenter's own prior turns in the thread (not the full thread), explicitly reasoned in
`02_data_annotation.ipynb` cell 15 as keeping context style/sentiment consistent with the scored
comment, at the acknowledged cost of missing same-side input from other commenters.
**Structural caveat to carry into Limitations [VERIFIED]:** thread size is structurally correlated
with a thread's apparent convincing rate (one delta per thread roughly fixed; negative examples
scale with thread branching) — the author flags this herself as relevant to any OP-level
aggregation, notably in `03.3_feature_engineering_op_relation.ipynb`.
**Figures:** two embedded figures exist in `02_data_annotation.ipynb` (cells 16, 31) — **[MISSING]
content**, image data was not extracted during this review; open the notebook directly to describe
and re-export them if they're wanted for the report.

### 3.2 Feature engineering
**Cover, as a "tried → rejected → replaced" narrative** (a genuinely useful rigor story for a
Methods section — each rejection is diagnosed, not just discarded):
- Sentiment: TextBlob (word-level, rejected as unsuited to long text) → VADER (rejected: conflates
  negative *topic* with negative *tone*) → **Gemini LLM scoring, tone-only prompt** (kept).
- Tone: RoBERTa emotion classifier (rejected: two-thirds collapse to "neutral") → **Gemini LLM**
  (kept: categories Logical/Passionate/Humorous/Sarcastic/Neutral/Other).
- Style: zero-shot `bart-large-mnli` (rejected: visible mislabeling on spot checks) → **Gemini LLM**
  (kept).
- Rhetorical appeals (ethos/pathos/logos): Empath lexicon (rejected: topic-driven, not structural)
  → **Gemini LLM, 0–1 scale** (kept; `logos_llm` shows the widest univariate split of any feature
  in this stage).
- Other kept features: adjective-adverb ratio (spaCy POS), readability (Flesch-Kincaid, via
  `textstat`), evidence-marker count (**explicitly adapted from Priniski & Horne's own keyword-stem
  method/code** — a direct, citable code-reuse instance for the References section), a
  HuggingFace persuasive-language detector (`chreh/persuasive_language_detector`), document
  embeddings (`BAAI/bge-base-en-v1.5`, PCA'd to ~187 components at ≥85% variance).
- Comment↔original-post relational features ("match"/"diff" pairs) computed in
  `03.3_feature_engineering_op_relation.ipynb`.
**All [VERIFIED]** from `03.1`–`03.3` and `common_functions.py`.
**Rejected-features bookkeeping [VERIFIED]:** 6 features formally flagged **untrusted**
(`sentiment_vader`, `tone_label`, `style_label`, `ethos_score`, `pathos_score`, `logos_score`),
persisted to `results/03_untrusted_features.json`, deliberately *kept in the data* (not deleted) so
`04.1_model_training.ipynb` could test their marginal value empirically rather than exclude them by
assumption.
**Implementation detail for Methods [VERIFIED]:** all Gemini-calling infrastructure (client setup,
retry/timeout handling, checkpointed batch scoring, prompt templates) is centralized in
`common_functions.py`, reused unchanged across `03.x`, `05`, and `08` — a genuine reproducibility
and consistency strength worth naming explicitly.
**Figure/table candidate [MISSING content, exists in notebook]:** a full numeric feature
correlation heatmap at the end of `03.3` ("Feature Correlation Heatmap (w/o embedding features)") —
exists as an embedded image, content not readable from text extraction; re-render from the
notebook if wanted.

### 3.3 Model training and honest-evaluation methodology
**Cover:** thread-grouped `GroupShuffleSplit` (80/20 by `thread_id`, zero thread overlap between
train/test, verified by assertion in the notebook) to prevent leakage; PR-AUC chosen and justified
as the primary metric over accuracy/ROC-AUC on a ~16% positive-rate problem; a validation grid of
3 classifiers (Logistic Regression, Random Forest, XGBoost) × 3 resampling strategies (none,
RandomUnderSampler, SMOTE) × 3 feature-set variants (trusted-only, trusted+untrusted,
embeddings-only) = 27 combinations, selected on an inner validation split, evaluated once on the
held-out test set. **[VERIFIED]**, `04.1_model_training.ipynb`.
**Figure [exists, content unknown]:** PR curve + ROC curve for the selected pipeline, side by side
— embedded image, not re-extracted here.

### 3.4 Baseline investigation and the length confound
**Cover, as the project's central methodological turning point:** a minimal TF-IDF+Logistic
Regression baseline (no engineered features at all) outperforming the full engineered pipeline;
feature-family ablation isolating which families help/hurt; TF-IDF coefficient inspection revealing
top predictive tokens are stopwords/length-adjacent, not rhetorical content; the discovery that
raw `word_count` alone outperforms every other model tried; within-thread robustness checks ruling
out a topic/thread-size artifact; the decision to fold `word_count` into the engineered feature set
rather than treat length as noise. **[VERIFIED]**, `04.2_baseline_comparison.ipynb` — see exact
numbers in `RESULTS_SHORTLIST.md` finding #1.
**Methods framing worth stating directly [INTERPRETIVE, built on verified facts]:** this sequence
(baseline beats main model → diagnose why → fix it → re-measure) is itself a demonstration of
rigorous empirical methodology, not a detour — worth narrating as such rather than only reporting
the final numbers.

### 3.5 Length-controlled analysis
**Cover:** two resampling strategies built specifically to break the length confound (a loose
word-count floor and a strict 1:1 length-decile match); re-measurement of every model's lift over
its own no-skill baseline on each sample; and an explicit worked example of a small-validation-set
overfitting failure caught by re-running model selection from scratch on the length-matched sample.
**[VERIFIED]**, `04.3_length_controlled_analysis.ipynb`. The overfitting example (re-selected
model scores *below* its own no-skill baseline on test, while reusing the original architecture
scores positively) is a strong, concrete pedagogical point for the Discussion's "lessons learned"
material — **[INTERPRETIVE]** framing choice, numbers themselves verified.

### 3.6 Rhetorical-strategy scoring (the project's main contribution)
**Cover:** the 8-strategy rubric (justification, concession, evidence, refutation, personal
narrative, call-to-action, simplification, question-asking), scored 1–5 by Gemini following the
*method* of Labruna et al. (2026), adapted from their 6 propaganda-technique strategies on news
text to 8 new strategies purpose-built for CMV; the prompt-design iteration from a v1 (comment-only
context) to v2 (batched by original post, so the LLM actually sees the argument being
refuted/conceded/etc.) — a concrete, quantified example of why this mattered is available (row
2421-style case studies). **[VERIFIED]**, `05_rhetorical_scoring.ipynb`.
**Validation [VERIFIED], explicitly self-described by the author as a lightweight check, not a
formal reliability study:** test-retest self-consistency reaches 95–99% exact-or-within-1-point
agreement across all 8 strategies; hand-scoring against one human judge (N=18) is weaker overall
and notably weakest on exactly the two strategies — `refutation` and `justification` — that later
turn out to matter most in Results. State this pairing explicitly; it is a real caveat on the
headline finding, not a side note.
**Denominator note [VERIFIED, important to get right]:** rhetorical scoring restricts to comments
≥30 words (or any length if convincing) — 2,753 comments, not the full 3,971. Any percentage from
`05`/`06`/`07` (e.g. a 23.3% convincing rate) is against this smaller population and is **not**
directly comparable to the 16.1% baseline from §3.1 without saying so.
**Out of scope for now:** `refutation_reannotation/` (a second, two-judge reannotation sample for
`refutation` scoring) is explicitly out of scope for this round of work per user instruction — do
not reference it, cite it, or treat it as pending work in this section. See
`READINESS_CHECKLIST.md`.

### 3.7 Strategy ablation
**Cover:** adding the 8 strategy scores to the existing engineered model for a real, order-robust
PR-AUC gain; a methodological finding that must be disclosed regardless of direction — the
underlying XGBoost+SMOTE pipeline is sensitive to column *order* on this correlated, 200+-column
feature matrix, large enough to rival some of the effects under study, root-caused to XGBoost's
tie-breaking behavior on near-equal split gains; the resulting convention of reporting every
PR-AUC from this point on as a mean±std over multiple column orderings. **[VERIFIED]**,
`06_strategy_ablation.ipynb`. This order-sensitivity disclosure is itself a notable piece of
methodological transparency worth describing explicitly in Methods, not only applying silently.
**Family-by-family and length-controlled retention results:** see `RESULTS_SHORTLIST.md` finding #3.

### 3.8 Strategy interpretation
**Cover:** SHAP beeswarm analysis and Mann-Whitney U rank-biserial effect sizes (model-free,
FDR-corrected across the 8 strategies), computed on the full population and again on the strict
length-matched sample, to separate which strategies' apparent importance is genuine vs.
length-confounded. **[VERIFIED]**, `07_strategy_interpretation.ipynb`. See
`RESULTS_SHORTLIST.md` finding #4 for the headline result (refutation).
**Figures [exist, content unknown]:** SHAP beeswarm plots (full-population and length-matched) —
embedded images, not re-extracted; `plot_category_histogram` convincing-rate-by-score panels for
each of the 8 strategies.

### 3.9 External validation
**Cover:** the Winning Arguments corpus (Tan et al. 2016, via `convokit`), an independent dataset
built with a different negative-sampling methodology (topically-matched pairs, not this project's
thread-leaf sampling); the same rubric, prompt, and the same **model architecture and training
configuration already selected** on the home dataset (`RobustScaler` + `SMOTE` + `XGBoost`, from
`04.1`/`04.2`, carried unchanged through `06`/`07`) — reused as-is, with **no re-selection of
architecture or configuration on WA** ("no feature selection or tuning against WA").

**Training/evaluation procedure — state this precisely, it is not zero-shot:** a **new model
instance is fit on WA's own native training split** and evaluated on WA's own native heldout
split (the same train/test partition the corpus ships with and the literature's own benchmark
numbers are computed on) — **[VERIFIED]**, `08_wa_external_validation.ipynb` cell 19 ("Fit on WA's
own native `train` split... evaluated on WA's own native `heldout` split... with no re-selection on
WA"). This is **reusing a selected architecture/configuration and fitting new parameters on WA's
training data** — distinct from (a) selecting/tuning a *new* configuration using WA data (not
done — the validation grid from `04.1` was not rerun against WA), and from (b) zero-shot transfer
without refitting (not done here — that is a separate, out-of-scope check, see below).

The explicit 5-question validation framing the notebook poses for itself is **[VERIFIED]**. See
`RESULTS_SHORTLIST.md` finding #5 for the headline (partial, not full, replication) result.

**Out of scope for the official report/presentation:** `wa_quick_experiments.ipynb` is a separate,
self-described scratch/exploratory notebook ("not documented to the same standard, and not
necessarily headed for the final report"). One of its three checks ("Experiment 2") *is* genuine
zero-shot transfer — the home-trained model applied directly to WA's heldout split with **no
refitting at all** — which is why it must not be confused with the §3.9 procedure above. Per
explicit instruction, this notebook and all three of its findings (embeddings on top of the robust
core; the zero-shot check; re-adding `concession`) are **excluded from the official report and
presentation** — retained only as a labeled, out-of-scope entry in `RESULTS_SHORTLIST.md` for
completeness of the research record, not as reportable results.

---

## 4. Results

**Cover:** this section should present, in order, the quantitative backbone already extracted in
`RESULTS_SHORTLIST.md` — recommend following that document's finding order (length confound →
length-controlled survival → strategy ablation gain → refutation/justification interpretation →
external-validation partial replication), each with its exact numbers, its source table, and its
figure if one exists. Do not re-derive numbers here; cross-reference `RESULTS_SHORTLIST.md`.

**Visualizations needed [MISSING as standalone files]:** every figure referenced across §3 above
exists only as an embedded, non-exported notebook output (confirmed: no PNG/SVG/JPG files exist
anywhere in the repo outside the two `previous work/` PDFs). Before the report can include any
figure, it must be re-rendered or screenshotted directly from the live notebook that produced it.
This is a single recurring gap, not eight separate ones — flagged once here, applies throughout.

---

## 5. Discussion

**Cover:**
- **Conclusions:** comment length is the dominant single signal in this dataset, but a real,
  length-independent rhetorical signal exists and survives rigorous length control; that signal
  concentrates specifically in `refutation`, with `justification`'s apparent importance mostly
  length-confounded — a strategy-level, not just feature-group-level, conclusion.
  **[VERIFIED→INTERPRETIVE synthesis]**, built directly on `07`'s own stated summary.
- **Insights:** the repeated pattern of catching and correcting an initial result (the length
  confound in 04.2, the small-validation-set overfitting failure in 04.3, the column-order
  sensitivity in 06) is itself a methodological insight worth naming explicitly — rigor as a
  finding, not just a precondition for one. **[INTERPRETIVE]**.
- **Deep-learning techniques used:** name them explicitly here even though they weren't the star
  of the story — Gemini LLM scoring (the project's main feature-generation technique),
  transformer-based sentence embeddings (BAAI/bge-base-en-v1.5), HuggingFace transformer
  classifiers evaluated and in two cases rejected (RoBERTa emotion, BART-MNLI zero-shot) in favor
  of LLM scoring, and a transformer-based persuasive-language detector that was kept. **[VERIFIED]**.
- **Limitations:** single subreddit/platform; English-only; LLM-scored features inherit the LLM's
  own calibration gaps, most acutely on exactly the two strategies (`refutation`, `justification`)
  that matter most for the headline conclusion; external validation shows the *structural* finding
  generalizes but the *specific-strategy* finding does not, with only a flagged-as-untested
  hypothesis for why; sample sizes at the top end of several strategies' score distributions are
  thin (e.g. n=8 for call-to-action at score 5); most point-estimate comparisons in this project
  have overlapping bootstrap confidence intervals and should be read as best-point-estimates, not
  statistically decisive rankings — a caveat the author herself repeats throughout 04.2/04.3.
  **[VERIFIED]**. (`refutation_reannotation/` is out of scope for now per user instruction — not
  listed as a limitation here.)
- **Suggested improvements:** testing the author's own flagged (and explicitly labeled untested)
  hypothesis about negative-sampling differences explaining the WA reversal; extending external
  validation to a third, differently-constructed corpus. **State these as proposed future work,
  not completed work** — per this task's constraints, do not write them as if already done.

---

## 6. References

**Cover, as separate items:**
- The four related-work/method papers in §2.3 (all four now **[VERIFIED]**, including page numbers).
- **Code reuse to cite explicitly:** Priniski & Horne's `collect_posts.py`/`discussion_analysis.py`
  (github.com/jpriniski/CMV) — both the data-collection approach and the evidence-marker keyword
  method were adapted from this code. **[VERIFIED]**.
- **Systems/libraries used:** PRAW (Reddit API), Google `google-genai` SDK (Gemini), HuggingFace
  Transformers (several named models, two rejected and two kept — see §3.2), spaCy, `textstat`,
  XGBoost, `imbalanced-learn` (SMOTE), SHAP, ConvoKit (Winning Arguments corpus access). **[VERIFIED]**.
- **AI tools used:** this is an explicit university requirement (item 6 as given: "References,
  including relevant existing code, systems, transfer learning and AI tools used"), not a
  journal-specific add-on — confirmed directly against the instructions given for this report. The
  research-method side is already fully in scope and documented above: Gemini, used extensively as
  a scoring tool throughout feature engineering and rhetorical-strategy scoring. **Remaining open
  question (narrower than "whether to disclose"):** whether this same References line should also
  name any AI coding assistant used while building the pipeline/notebooks themselves, alongside
  Gemini's research-method role. See `READINESS_CHECKLIST.md`.

---

## 7. Link to code and data

**Cover:** point to this repository (branch `final-project`), the notebook sequence in `this work/`
in execution order (README.md's own table is already a ready-made roadmap), and the data-access
instructions already written in `README.md` (Google Drive raw-data links; Winning Arguments corpus
fetched automatically via `convokit`). **[VERIFIED]** — this section can largely reuse existing
README content, adapted to the report's citation style, once a confirmed state of the repo is tied
to a specific commit/release for reproducibility. **Decision needed:** whether to tag or cite a
specific commit, since the repo is still actively being edited (this very task adds files to it).
