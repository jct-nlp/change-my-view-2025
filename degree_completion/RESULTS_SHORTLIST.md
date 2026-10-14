# Results Shortlist

Eight findings, selected for scientific importance and narrative coverage — not for which numbers
look most impressive. Three are explicitly non-"flattering" (a confound that undercuts the main
model, a reversal under external validation, a reliability caveat on the two strategies that matter
most) because the task calls for including meaningful negative/inconclusive findings, and because
an honest report is a stronger report. All numbers below were read directly from notebook
execute/stream output by independent research passes over the source notebooks; exact values are
quoted, not estimated.

**Note on Finding 6:** as of 2026-10-14, Finding 6 (`wa_quick_experiments.ipynb`'s zero-shot
cross-corpus check) is explicitly **out of scope for the official report and presentation**,
per instruction. It stays in this document only to preserve the research record — not as a
proposed slide, backup slide, or report result. See Finding 6 itself for the full status note.

---

## Finding 1 — A length confound: the engineered feature pipeline is beaten by raw word count

**Finding:** A minimal TF-IDF+Logistic-Regression baseline, using none of the project's engineered
features, outperforms the full engineered-feature model on the same held-out test set. A
single-feature model using only `word_count` beats both.

**Quantitative results (test set, 709 comments / 81 threads, no-skill PR-AUC = 0.161):**
| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Selected engineered pipeline (XGB+SMOTE, trusted+untrusted features) | 0.3848 | 0.6982 | 0.4046 (tuned threshold) |
| TF-IDF + Logistic Regression (raw text only) | 0.5119 | 0.8150 | 0.1235 |
| `word_count` alone (single feature, Logistic Regression) | 0.5256 | 0.8294 | 0.2105 |

`word_count` correlates with `is_convincing` at r=0.3716 (p=2.92e-130); mean word count is 78.0 for
non-convincing comments vs. 216.7 for convincing ones. Within-thread check (controls for
topic/thread-size effects): in 200 threads containing both classes, the convincing comment was
longer than that thread's own non-convincing average in 153/200 (76.5%, Wilcoxon p=4.15e-19).
Folding `word_count` into the engineered feature set and re-running model selection from scratch
closes most of the gap: **test PR-AUC 0.4922** (vs. the original 0.3848), with bootstrap 95% CIs
barely overlapping the original selection — the most statistically decisive single comparison in
this notebook pair.

**Source:** `this work/04.2_baseline_comparison.ipynb` (TF-IDF baseline, word-count discovery,
re-selection with word_count); cross-referenced against `04.1_model_training.ipynb` (original
selected-pipeline numbers).

**Figure/table:** `word_count` histogram by `is_convincing` (binwidth 50, comments <600 words) —
embedded matplotlib output, not exported as a standalone file; TF-IDF coefficient-inspection table
(top predictive tokens, mostly stopwords and the token `edit`).

**Why it matters:** this is the project's central methodological pivot. It demonstrates that the
original feature-engineering effort was, on its own, being outcompeted by a confound the modeling
pipeline hadn't accounted for — and that catching this changed the rest of the project's direction
(length-controlled analysis, and later testing whether rhetorical strategies are "just length in
disguise").

**Limitations/qualifications:** most point-estimate PR-AUC comparisons in this project have
overlapping bootstrap 95% confidence intervals at this test-set size (709 comments); the
word_count-vs-engineered gap is the one comparison here with CIs that barely overlap, so treat it
as the most, not the only, solid point. The selected pipeline's exact PR-AUC is also sensitive to
feature-column order (see Finding 3) — report it as a range, not a single point estimate, if
precision matters for the claim being made.

**Presentation suitability:** **Yes, strong candidate.** Concrete, intuitive, and sets up every
later finding. Recommend one slide, framed as "the surprising result that changed the project's
direction," not a buried caveat.

---

## Finding 2 — The length-controlled signal collapses but does not vanish

**Finding:** Once comment length is matched between classes (removing the confound from Finding
1), every model's apparent skill drops by roughly 75–90% — but none drop to zero or negative. A
small, genuine, length-independent signal survives in every approach tried.

**Quantitative results (lift over the test fold's own no-skill baseline, computed correctly — an
earlier draft of this analysis had a baseline-computation error, explicitly caught and fixed in
the notebook itself):**
| Sample | TF-IDF lift | word_count lift | Engineered-pipeline lift |
|---|---|---|---|
| Full dataset | 0.3088 | 0.3225 | 0.1817 |
| Loose length floor (1,638 comments) | 0.0478 | 0.0898 | 0.0165 |
| Strict 1:1 length-matched (1,235 comments) | 0.0767 | 0.0576 | 0.0334 |

A separate, explicitly reported methodological failure in the same notebook: re-running the full
27-combination model-selection grid from scratch on the small (1,235-row) length-matched sample
selects a *different* architecture that scores **below** its own no-skill baseline on test
(lift = −0.0162), a textbook small-validation-set overfitting case (205 validation rows, 27
candidates compared) — while simply reusing the architecture already selected on the full dataset
scores positively on the same test set (lift = +0.0334).

**Source:** `this work/04.3_length_controlled_analysis.ipynb` (both the main length-control table
and the re-selection overfitting example).

**Figure/table:** no chart objects beyond reused tables; this section is primarily tabular.

**Why it matters:** this is the finding that justifies everything downstream (strategy ablation,
interpretation) being worth doing at all — it establishes that there is real non-length signal to
go looking for, while being honest that most of each model's apparent skill really was about
length. It also doubles as a clean worked example of a methodological pitfall (re-selecting a
model on a too-small sample) worth including on its own merits.

**Limitations/qualifications:** the strict length-matched sample is small (1,235 comments); the
author explicitly frames the re-selection failure as a lesson about validation-set size, not
evidence against model re-selection as a practice in general — don't over-generalize it.

**Presentation suitability:** **Partially — use the headline table, skip the overfitting aside.**
The three-sample lift table is clear and important; the re-selection-failure story is valuable for
the written report's Discussion/lessons-learned material but is a methodological detour that would
cost more time than it's worth in a 15-minute talk.

---

## Finding 3 — Rhetorical strategy scores add signal beyond length, unlike every other feature family

**Finding:** Adding the 8 LLM-scored rhetorical strategies to the existing 229-feature engineered
model produces a real, order-robust PR-AUC gain — and this gain survives length control far better
than any other feature family tested, including length itself.

**Quantitative results:**
- Full population: current model alone 0.4467±0.0196 PR-AUC → + 8 strategy scores 0.5202±0.0177
  (**+0.0735**, clears the order-noise band). Strategy scores alone: 0.3538±0.0079 (real
  standalone signal, between no-skill 0.2382 and the full model).
- Best single combination found anywhere in the modeling work: **8 strategy scores + `word_count`
  alone (9 columns) = 0.5379±0.0109 PR-AUC**, beating every larger combination tested, including
  the full 237-column feature set.
- Length-controlled retention (strict 1:1 length-matched sample, 1,235 comments): the *strategy
  gain itself* retains **84%** of its full-population value (0.0664→0.0557), while `word_count`'s
  own lift retains only ~18% and the full engineered pipeline's only ~25% (consistent figures
  reported in both `04.3` and `06`).

**Source:** `this work/06_strategy_ablation.ipynb`.

**Important methodological disclosure attached to this finding:** the underlying XGBoost+SMOTE
pipeline on this correlated, 200+-column feature matrix is sensitive to column *order* — refitting
identical data with only column order shuffled swings PR-AUC by as much as 0.0422, large enough to
rival some of the effects under study. Root-caused to XGBoost's deterministic tie-breaking among
near-equal split gains on correlated features (not a SMOTE artifact). Every PR-AUC number from
this notebook onward is reported as a mean±std over multiple column orderings specifically because
of this — a real reliability caveat on the precision of every number in this and the next finding,
not just a footnote.

**Why it matters:** this is the finding that the whole rhetorical-scoring effort (notebook 05) was
built to test, and it is a clean, unambiguous result in a research project that otherwise has many
overlapping-CI, hard-to-call comparisons — the strategy-score gain is large enough and clears the
order-noise band in a way most other comparisons in this project do not.

**Limitations/qualifications:** always report PR-AUC as mean±std over orderings, never a single
run's value, for any number from this notebook or `07`.

**Presentation suitability:** **Yes, strong candidate — the project's clearest "yes" result.**
Recommend pairing the headline gain number with one sentence on the order-robustness convention,
since a sharp examiner question ("how stable is that number?") is very likely and this notebook
already has the answer prepared.

---

## Finding 4 — Which strategy carries the signal: refutation is robust, justification is mostly a length proxy

**Finding:** Of the 8 scored rhetorical strategies, `refutation` is the one that holds up most
consistently once length is controlled for — ranking highest or near-highest by both SHAP
importance and a model-free (Mann-Whitney) effect size, on both the full population and the
length-matched sample. `justification` shows the single largest *raw* effect size, but its SHAP
contribution collapses once length is controlled, indicating most of its apparent importance is
length-confounded rather than independently predictive.

**Quantitative results:**
- SHAP mean |value|, full population: word_count 1.0245 » **refutation 0.6034** » justification
  0.4156 > evidence 0.3592 > concession 0.3439 > simplification 0.3193 > personal_narrative 0.1853
  ≈ call_to_action 0.1837 > question_asking 0.1518 (least influential).
- SHAP mean |value|, strict length-matched sample: word_count 0.8402 » **refutation 0.5193**
  (holds rank, modest decline) > concession 0.3445 > simplification 0.3383 > evidence 0.3090 >
  call_to_action 0.2707 > personal_narrative 0.2168 > **justification 0.1956** (collapses from 3rd
  to 7th-of-8 — most of its full-population importance was length-related) > question_asking 0.1806.
- Mann-Whitney rank-biserial effect size, full population (FDR-corrected): justification r=0.3584
  > refutation r=0.3149 > simplification r=0.1944 > call_to_action r=0.1584 > concession r=0.1481 >
  evidence r=0.1018 > question_asking r=0.0069 (not significant) > personal_narrative r=−0.0217
  (not significant).
- Same test, length-matched: only 3 of 8 strategies remain significant — refutation (r=0.154, 49%
  of full-population effect retained), justification (r=0.131, 37% retained), simplification
  (r=0.079). `personal_narrative` is the one strategy that *gains* significance under matching,
  flipping to a small, reliable *negative* association (r=−0.086).

**Source:** `this work/07_strategy_interpretation.ipynb`.

**Why it matters:** this narrows Finding 3's group-level result ("strategy scores, as a block,
survive length control") down to a specific, falsifiable claim about which individual strategy
drives it — the kind of precision a report should aim for instead of stopping at "rhetoric
matters."

**Limitations/qualifications — important, do not omit:** `refutation` and `justification` are
also the two strategies that `05_rhetorical_scoring.ipynb`'s own validation flagged as hardest for
the LLM to calibrate against human judgment (lowest exact-match agreement in both the test-retest
and the hand-scoring checks). The strongest univariate signal in this whole analysis arm comes from
the two most noisily-scored strategies — a caveat that belongs directly next to this finding, not
filed separately under "limitations" where a reader might miss the connection.

**Presentation suitability:** **Yes, strong candidate — the project's most specific and
examinable finding.** Recommend showing the two SHAP rankings side by side (full vs.
length-matched) as the single most visually clear way to make the point.

---

## Finding 5 — External validation: the structural finding generalizes, but which strategy dominates reverses

**Finding:** On the independent Winning Arguments (WA) corpus — built with a different
negative-sampling method (topically-matched pairs, not this project's thread-leaf sampling) — the
rhetorical-strategy scores again add predictive value beyond `word_count` and again partly survive
length control. But the specific strategy that dominates is different: `justification`, not
`refutation`, is WA's top strategy by both SHAP and effect size.

**Procedure — state this precisely, it is not zero-shot transfer:** the model architecture and
training configuration already selected on the home dataset (`RobustScaler` + `SMOTE` + `XGBoost`,
chosen in `04.1`/`04.2`, reused unchanged through `06`/`07`) was carried over as-is — no new
architecture/configuration selection was performed on WA. A **new model instance was then fit on
WA's own native training split and evaluated on WA's own native heldout split**
(`08_wa_external_validation.ipynb` cell 19, verified directly). In short: reuse the selected
configuration, refit its parameters on WA's training data, evaluate on WA's held-out data — distinct
from tuning a new configuration on WA (not done) and distinct from zero-shot transfer without
refitting (not done here; that is a separate, out-of-scope check — see Finding 6).

**Quantitative results (WA corpus: 8,106 comments, no-skill baseline 0.5291; word_count confound
weaker here, r=0.1516 vs. 0.3716 at home):**
| Feature set | PR-AUC | lift over no-skill |
|---|---|---|
| word_count only | 0.5767 | 0.0476 |
| strategy scores only | 0.5967 | 0.0676 |
| robust 4-feature core (refutation, justification, simplification, word_count) | 0.5919 | 0.0628 |
| best found ("−call_to_action", 8 cols) | 0.5937 | — |

Pairwise accuracy (comparable to the WA literature benchmark, 805 heldout pairs): word_count only
55.59%; strategy scores only 57.27%; robust core 58.68%; best found (−call_to_action) 59.10%.
Literature reference points (Labruna et al. 2026): "Independent Scoring" baselines 53.78–56.66%,
full MS-PS method 60.59–64.53%. This project's simple 4-feature robust core sits above the
independent-scoring baselines and below the full literature method — a reasonable, honestly
bounded place to land, explicitly **not** claimed as a like-for-like beat of the literature.

**The reversal, stated precisely:** on the home dataset, `refutation` is top by SHAP (0.603 mean
|SHAP|) and 2nd by effect size. On WA, `justification` is top by both SHAP and Mann-Whitney effect
size (r=0.1962), while `refutation` is 4th by effect size and near-bottom by SHAP. Length control
on WA still leaves `justification`'s effect 84% intact (unlike at home, where it collapsed) — this
is a genuine divergence, not a statistical-noise artifact, since WA's test-retest reliability for
every strategy (including refutation, 76.5% exact match) is *better* than at home, ruling out
"scoring noise" as the explanation.

**The author's own proposed explanation — explicitly labeled a hypothesis, not a tested finding:**
the reversal may stem from how the two datasets' negative examples are selected — WA's losing
replies are Jaccard-matched to be topically similar and substantive to the winner, while this
project's non-convincing comments are not filtered for similarity at all. This is offered with one
supporting (not proving) number — 36.6% of the home dataset's unfiltered non-convincing pool is
under 30 words, vs. a much narrower winner/loser length gap on WA — but no experiment in this
project tests it directly. **State this as an open hypothesis in the report, explicitly
attributed as such, not as an established explanation.**

**Source:** `this work/08_wa_external_validation.ipynb`.

**Why it matters:** honest external validation is exactly the kind of result that distinguishes a
rigorous project from an overfit one — the structural claim ("rhetorical strategies carry
length-independent signal") replicates cleanly; the more specific claim ("refutation is *the*
strategy") does not, and the project reports that plainly rather than cherry-picking the result
that would have looked cleaner.

**Limitations/qualifications:** this is a partial, not full, replication by the notebook's own
framing — don't round it up to "the finding generalized."

**Presentation suitability:** **Yes — and should be framed as a strength, not hedged.** Examiners
tend to reward honest reporting of a partial replication over a project that only shows results
that confirm its own earlier claims. One slide: "it generalizes structurally; the details differ;
here's an open hypothesis for why."

---

## Finding 6 — EXCLUDED FROM OFFICIAL DELIVERABLES — cross-corpus zero-shot transfer works above chance but degrades

**Status (2026-10-14): explicitly out of scope for the official report and presentation**, per
direct instruction. Retained here only so this piece of the research record isn't lost — not as a
candidate for the report, slides, or backup slides. Do not include it, reference it, or draw on it
when drafting any of those deliverables.

**Finding:** A model trained only on the home dataset, applied directly to WA's held-out split with
**no refitting at all** — genuine zero-shot transfer, unlike the main WA validation in Finding 5,
which refits a new model on WA's own training split — performs above chance but clearly below the
Finding-5 (refit) version.

**Quantitative results:** zero-shot transfer — PR-AUC 0.5621 (vs. 0.5291 no-skill), ROC-AUC 0.5453,
pairwise accuracy 53.29%. Finding 5's WA-refit version: pairwise accuracy 58.09–58.68%.

**Source:** `this work/wa_quick_experiments.ipynb` — self-described in its own opening cell as a
scratch/exploratory notebook, "not documented to the same standard, and not necessarily headed for
the final report." That self-description is why it's excluded here, not just a caveat on how to
present it.

**Why it matters (research-record note, not a reporting instruction):** a clean, genuine
negative/limited-generalization result: "some of what the model learned carries over, but letting
it see the target data still helps meaningfully" — relevant to internal understanding of how far
this project's findings generalize without refitting, even though it isn't going in the deliverables.

**Presentation suitability:** **Excluded — not a main slide, not a backup slide.** Per instruction,
this is out of scope for the presentation entirely, including backup slides. If asked an
examiner question about zero-shot generalization specifically, this is background knowledge to
have in mind, not material to put on screen.

---

## Finding 7 — Rubric validation: strong self-consistency, weaker human agreement, concentrated exactly where it matters most

**Finding:** The 8-strategy rubric's LLM scores are highly self-consistent on repeat scoring, but
agree less well with a single human judge's hand-scoring — and the two strategies with the weakest
human agreement are `refutation` and `justification`, the same two strategies Finding 4 identifies
as the most predictively important.

**Quantitative results:** test-retest self-consistency (N=200): all 8 strategies reach 95–99%
within-±1-point agreement; exact-match ranges from 63.8% (`refutation`, weakest) to 86.9%
(`question_asking`, strongest). Hand-scoring vs. one human judge (N=18, before the judge's own
corrections after reviewing disagreements): exact-match ranges from 22.2% (`justification`,
weakest) and 27.8% (`refutation`) up to 77.8% (`question_asking`/`personal_narrative`). After the
judge reviewed and corrected her own first-pass scores, agreement rose across the board (e.g.
justification 11.1%→22.2%, refutation 16.7%→27.8%) but `refutation`/`justification` remained the
two hardest to calibrate (mean absolute difference ~0.89, the highest of the 8).

**Source:** `this work/05_rhetorical_scoring.ipynb`.

**Why it matters:** this is a direct, important qualification on the project's headline finding
(Finding 4) — the strategy with the strongest predictive signal is also the one the measurement
instrument is least reliable at. It does not invalidate Finding 4, but it bounds how strongly that
finding should be stated. (`refutation_reannotation/` is out of scope for now per user instruction
— not referenced here as a remedy or follow-up.)

**Limitations/qualifications:** the hand-scoring check is explicitly self-described by the author
as "a lightweight check, not a formal reliability study" (small N, one judge) — don't present it as
a rigorous inter-annotator-agreement study in the report; present it as what it is.

**Presentation suitability:** **Yes, but as one caveat sentence attached directly to Finding 4's
slide, not a separate slide.** It's too important to omit, but doesn't carry enough independent
weight for its own 1–2 minutes in a 15-minute talk.

---

## Finding 8 — A 9-feature model matches almost all of the full pipeline's predictive power

**Finding:** The single best feature combination found anywhere in this project — 8 strategy
scores plus `word_count` (9 columns total) — outperforms the full 237-column engineered-plus-
strategy feature set, and does so with a small, interpretable, directly-explainable set of inputs.

**Quantitative results:** 9-column combination: 0.5379±0.0109 PR-AUC vs. the full 237-column set's
0.5202±0.0177 — a small model beating a much larger one, not a close second. (Same number already
appears in Finding 3; listed again here because it supports a distinct point.)

**Source:** `this work/06_strategy_ablation.ipynb`.

**Why it matters:** for a report evaluated partly on "deep-learning techniques, limitations, and
suggested improvements," this is a useful counterpoint to a narrative that bigger/more features
are always better — a concrete, numeric argument for parsimony and interpretability.

**Limitations/qualifications:** same order-sensitivity caveat as Finding 3 applies (reported as
mean±std, not a single point estimate).

**Presentation suitability:** **Optional — good backup slide, not essential to the core 15-minute
arc.** Use if time allows after Findings 1, 3, 4, and 5; otherwise fold the single number into
Finding 3's slide as a one-line addition ("the 9-feature version actually beats the full set").
