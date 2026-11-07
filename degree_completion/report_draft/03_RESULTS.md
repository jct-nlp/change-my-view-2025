# 3. Results

This chapter reports the project's quantitative findings, organized around the three research
questions introduced in Chapter 1: predictability and the role of comment length (RQ1), the
incremental value and importance of rhetorical-strategy scores (RQ2), and cross-corpus
generalization on an independently constructed dataset (RQ3). Procedural detail — how each
dataset, sample, or model was constructed — is not repeated here except where needed to interpret
a number; it is described in full in Chapter 2. Broader interpretation of these findings, their
relationship to prior work, and their limitations are reserved for the Discussion chapter.

## 3.1 RQ1: Predictability and Comment Length

### 3.1.1 Performance of the engineered-feature pipeline

Model selection on the inner validation split (§2.3) identified XGBoost with SMOTE oversampling on
the 239-feature trusted+untrusted representation as the best-performing combination, with
validation PR-AUC 0.3063 against that fold's own no-skill baseline of 0.1690. Evaluated once on the
709-comment held-out test set, this configuration reached **PR-AUC 0.3848** (ROC-AUC 0.6982, F1
0.4046 at a validation-tuned threshold), against a test-fold no-skill baseline of 0.2031. This is a
real improvement over no-skill, but it is the starting point the remainder of this section
re-examines, not the project's final answer to RQ1.

### 3.1.2 Comparison with TF-IDF and word-count baselines

Two substantially simpler baselines, evaluated on the identical 709-comment test fold, both
outperformed the selected engineered pipeline. A TF-IDF-vectorized bag-of-words representation with
logistic regression, using no engineered features at all, reached PR-AUC 0.5119. A single-feature
logistic regression using only the comment's raw word count reached PR-AUC 0.5256 — the strongest
of the three models, using the least information.

**Table N.** Held-out PR-AUC and ROC-AUC for the engineered-feature pipeline and the two baselines
that motivated this section, on the identical 709-comment test fold (no-skill PR-AUC 0.2031). The
engineered pipeline is the only order-sensitive row (§2.3, §2.7); its order-robust mean ± SD across
eight column-order shuffles is given alongside its single-fit point estimate. The other three rows
have no column-order degree of freedom and are reported as single-fit point estimates only.

| Model | PR-AUC | PR-AUC (order-robust mean ± SD) | ROC-AUC | ROC-AUC (order-robust mean ± SD) |
|---|---|---|---|---|
| Majority-class (no-skill reference) | 0.2031 | n/a (order-invariant) | 0.5000 | n/a (order-invariant) |
| Selected engineered pipeline (XGBoost + SMOTE, trusted+untrusted) | 0.3848 | 0.3676 ± 0.0182 | 0.6982 | 0.6903 ± 0.0096 |
| TF-IDF + Logistic Regression | 0.5119 | n/a (order-invariant) | 0.8150 | n/a (order-invariant) |
| Word count alone (1 feature) | 0.5256 | n/a (order-invariant) | 0.8294 | n/a (order-invariant) |

The "majority-class" row is this fold's own positive-class prevalence, not a property of a hard
always-negative decision rule (§2.3). The three non-reference values are plotted in Figure 1
(§2.4).

Inspecting the TF-IDF model's coefficients showed that the great majority of its most heavily
weighted tokens were common stopwords, together with the token "edit" — a pattern more consistent
with the model tracking how much text a comment contains than with it tracking rhetorical content.
A follow-up check, refitting TF-IDF with English stopwords removed, found that this did not close the
gap to `word_count`: PR-AUC fell from 0.5119 to 0.4394, and the correlation between the model's
predicted probability and raw word count only dropped from r = 0.8433 to r = 0.7417 — still far
above the raw label's own correlation with length (r = 0.3716, §3.1.3) — so stopword removal did not
cleanly separate a length-independent rhetorical signal from a length-proxy effect: the lost
predictive power is at least as consistent with stopwords being an unusually reliable length-tracking
channel — common function words appear in nearly every sentence and scale smoothly with length,
unlike sparser content words — as it is with them holding genuinely separate, non-length content.

Two further qualifications apply to this comparison. First, the engineered pipeline's single-fit
point estimate (0.3848) sits toward the high end of what the architecture actually produces across
column orderings — the order-robust mean in Table N (0.3676 ± 0.0182) is about one SD lower — since
XGBoost's tie-breaking among this feature set's many correlated columns is itself sensitive to
column order, with the mechanism set out fully in §2.3. Second, a
bootstrap-based reliability check (§2.3) gives 95% confidence intervals of [0.311, 0.463] for the
engineered pipeline and [0.430, 0.603] for TF-IDF, which overlap substantially. The ranking "word
count beats TF-IDF beats the engineered pipeline" is this test fold's best point-estimate ordering,
consistent across every variant of the comparison tried, but it is not a gap any single
709-comment split proves beyond its own sampling noise.

### 3.1.3 Comment-length distributions and within-thread evidence

Comment length and outcome are strongly associated in the full 3,971-comment dataset: convincing
comments average 216.7 words against 78.0 for non-convincing comments, a point-biserial correlation
of r = 0.3716 (p = 2.92 × 10⁻¹³⁰). This distribution, by outcome, is shown in Figure 4 (§2.4).

Two within-thread checks rule out this association being purely an artifact of some threads being
systematically longer or more active than others. A fixed-effects check, demeaning both word count
and outcome by thread before correlating them, found the relationship weakens but survives: r drops
from 0.3716 across the whole dataset to r = 0.2909 within threads (p = 2.78 × 10⁻⁷⁸) — some of the
raw correlation genuinely is thread-level, but a strong, highly significant relationship remains
after removing it. A paired comparison restricted to the 200 threads (of 401) containing at least
one comment of each outcome found the convincing comment longer than that same thread's own
non-convincing average in 153 of the 200 threads (76.5%), by a median of 87.4 words (Wilcoxon
signed-rank p = 4.15 × 10⁻¹⁹). Even comparing only within the same thread, on the same topic, the
comment that earns the delta is usually the longer one.

### 3.1.4 Length-controlled predictive comparisons

Two length-matched samples were constructed from the full 3,971-comment dataset to test whether
each model's apparent skill depends on length: a loose floor keeping every convincing comment and
restricting non-convincing comments to at least the convincing group's own 25th-percentile length
(84 words; 1,638 comments), and a strict decile-bin match sampling non-convincing comments from the
same length deciles as convincing comments in matching counts (1,235 comments). Because both
samples are themselves built from the outcome label and length values under study, their test
partitions are controlled, outcome-balanced evaluation samples rather than naturally distributed
held-out populations (§2.5). For each sample,
and for the full dataset, each model's PR-AUC lift over that sample's own test-fold no-skill
baseline — not the whole-sample rate, which can drift from the actual test-fold rate on a split
this small — was computed for TF-IDF, word count alone, and the engineered pipeline.

**Table N.** PR-AUC lift over each sample's own test-fold no-skill baseline, full dataset versus the
two length-controlled samples (`04.3_length_controlled_analysis.ipynb`).

| Sample | n | Test-fold no-skill | TF-IDF lift | `word_count` lift | Engineered-pipeline lift |
|---|---|---|---|---|---|
| Full dataset | 3,971 | 0.2031 | 0.3088 | 0.3225 | 0.1817 |
| Loose length floor | 1,638 | 0.3903 | 0.0478 | 0.0898 | 0.0165 |
| Strict decile-bin match | 1,235 | 0.4679 | 0.0767 | 0.0576 | 0.0334 |

Every model's lift collapses sharply once length stops separating the classes — by roughly
three-quarters to seven-eighths of its full-dataset value, depending on the model and the
strictness of the match — but in no case falls to zero or below. `word_count`'s own lift falls to
about 18% of its full-dataset value under the strict match (0.0576 / 0.3225), and the engineered
pipeline's lift falls to a comparable 18% (0.0334 / 0.1817). An order-robust check on the engineered
pipeline's own lift (mean over eight column orderings) moves these point estimates by at most
roughly one order-noise standard deviation, in either direction depending on the sample, and does
not change this qualitative pattern. Some predictive signal beyond raw length survives length
control for every approach tested, though its degree is modest in absolute terms and the samples
involved are small (1,235–1,638 rows).

Re-running the full 27-combination selection grid (§2.3) from scratch on the strict decile-matched
sample illustrates a distinct methodological risk rather than a substantive finding about length.
The grid's winner by validation PR-AUC on this sample (`RF (no resampling) | embeddings-only`, validation
PR-AUC 0.6656 against a 205-row validation set) scored PR-AUC 0.4518 on the corresponding test
partition — *below* that partition's own no-skill baseline of 0.4679, a lift of −0.0162. Simply
reusing the architecture already selected on the full dataset (XGBoost + SMOTE, trusted+untrusted),
without re-selecting, scored PR-AUC 0.5013 on the same test partition, a lift of +0.0334. This is a
textbook small-validation-set overfitting case — 27 candidate combinations compared against only
205 validation rows — not evidence that model re-selection is unsound as a general practice.

Separately from the length-matched resampling above, the engineered pipeline was re-selected on the
full dataset with `word_count` folded in as an additional candidate feature, via the same
27-combination grid used in §2.3. The same architecture won again (XGBoost + SMOTE,
trusted+untrusted, now 240 features including `word_count`), and test PR-AUC rose from 0.3848 to
**0.4922** — closing most of the gap to TF-IDF (0.5119). Order-robust means move in the same
direction (0.3676 ± 0.0182 to 0.4831 ± 0.0257), and the two configurations' bootstrap 95%
confidence intervals, [0.311, 0.463] and [0.410, 0.586], barely overlap. `word_count` is accordingly treated as a
first-class engineered feature in every analysis from §2.5 onward, not only as an external
reference point.

### 3.1.5 Interpreting the pivot

Taken together, these results motivated a change in the project's central question. Comment length
is associated with delta-award status more strongly than any individual engineered feature or
feature family tried, and much of the apparent predictive advantage of both the original engineered
pipeline and a plain bag-of-words model is attributable to this association rather than to
rhetorical content specifically. This is an empirical, associative finding: it establishes that
length is informative and that engineered features alone did not capture that information well, not
that length *causes* persuasive success, nor that every feature correlated with length is
redundant with it. The length-controlled comparisons in §3.1.4 show that a modest amount of
non-length signal survives in every model tested, which is the empirical basis for asking, in
§3.2, whether an explicitly defined set of rhetorical strategies can characterize that signal more
directly than engineered linguistic features did.

## 3.2 RQ2: Incremental Value and Importance of Rhetorical Strategies

### 3.2.1 Eight-strategy scoring and the evaluated population

Eight rhetorical strategies — justification, concession, evidence, refutation, personal narrative,
call-to-action, simplification, and question-asking — were scored on an ordinal 1–5 scale by an LLM
(Gemini), following the general methodology of Labruna et al. (2026) adapted to a taxonomy defined
independently for this project. Scoring was restricted to comments of at least 30 words, or of any
length if delta-awarded, yielding a scored population of **2,753 comments across 400 original
posts**. Because this restriction disproportionately removes short, non-convincing comments while
retaining every convincing comment, the scored population's convincing-comment prevalence (641 of
2,753, 23.3%) is substantially higher than the full dataset's 16.1%; percentages reported from this
point onward use this subset's own denominator unless stated otherwise.

Two reliability checks, both explicitly lightweight rather than a formal inter-annotator-agreement
study, accompany these scores. A test-retest check (200 comments scored twice) found 95–99% of
repeat scores within one point of the original across all eight strategies (mean absolute
difference 0.14–0.41). A comparison against a single human judge's hand-scoring of 18 comments (9
convincing, 9 not), after the judge reviewed her own first-pass disagreements with the LLM, found
at least 83.3% agreement within one point for every strategy (mean absolute difference 0.28–0.89).

**Table N.** Reliability figures for each strategy (definitions in Methods, Table N, §2.6), test-retest
(n = 200) and human agreement (n = 18 — the judge's final scores after reviewing her own first-pass
disagreements with the LLM), both at the same ±1-point tolerance (`05_rhetorical_scoring.ipynb`).

| Strategy | Test-retest: within ±1 pt | Test-retest: mean abs. diff. | Human agreement: within ±1 pt | Human agreement: mean abs. diff. |
|---|---|---|---|---|
| Justification | 98.5% | 0.29 | 88.9% | 0.89 |
| Concession | 99.0% | 0.29 | 94.4% | 0.67 |
| Evidence | 99.5% | 0.24 | 88.9% | 0.72 |
| Refutation | 95.5% | 0.41 | 83.3% | 0.89 |
| Personal narrative | 99.5% | 0.14 | 94.4% | 0.28 |
| Call-to-action | 95.5% | 0.30 | 100.0% | 0.67 |
| Simplification | 98.0% | 0.27 | 100.0% | 0.28 |
| Question-asking | 98.0% | 0.16 | 94.4% | 0.28 |

Refutation and justification are the two strategies with the weakest human-judge agreement (lowest
within-±1 rates and, tied, the highest mean absolute difference of the eight). As shown below, they
are also the two strategies most central to this project's predictive and interpretive findings —
a pairing that qualifies, without invalidating, the results that follow.

### 3.2.2 Engineered baseline versus baseline plus strategy scores

The eight strategy scores were appended to a 229-feature engineered baseline (the 228-feature
trusted representation with `word_count` added) and evaluated with the fixed XGBoost + SMOTE
architecture, restricted to the 2,753-comment scored population (thread-grouped split: 2,161 train
/ 592 test). Because of this pipeline's column-order sensitivity — reordering the same 237 columns
swung PR-AUC by as much as 0.0482 in a direct check (§2.3) — every PR-AUC below is an order-robust
mean ± SD across eight column orderings, not a single-order point estimate.

**Table N.** Strategy-score ablation on the 2,753-comment scored population (test-fold no-skill
PR-AUC 0.2382; `06_strategy_ablation.ipynb`).

| Feature set | PR-AUC (mean ± SD) | ROC-AUC (mean) |
|---|---|---|
| Engineered baseline (no strategy scores), 229 columns | 0.4434 ± 0.0187 | 0.7041 |
| Engineered baseline + 8 strategy scores, 237 columns | 0.5098 ± 0.0200 | 0.7657 |
| Strategy scores only, 8 columns | 0.3514 ± 0.0065 | 0.6733 |

[FIGURE: `figures/report/figure2_strategy_incremental_value.png` — **Figure N.** Held-out PR-AUC
for the engineered baseline with and without the eight strategy scores, and for the strategy scores
alone, on the 2,753-comment scored population (`06_strategy_ablation.ipynb`).]

Adding the eight strategy scores raises PR-AUC from 0.4434 to 0.5098, a gain of **0.0664**, which
clears the combined order-noise band of the two configurations. The strategy scores alone reach 0.3514 — a real,
standalone signal, positioned between the scored population's own no-skill baseline (0.2382) and
the combined model, but not a replacement for the existing engineered features.

### 3.2.3 Survival of the gain under length matching

Whether this gain reflects something beyond comment length, rather than another proxy for it, was
tested on a strict decile-bin length-matched sample built within the 2,753-comment scored
population itself — independently constructed from §3.1.4's full-dataset sample of the same size
(1,235 comments).

**Table N.** The strategy-score gain, full scored population versus the length-matched sample built
within it (`06_strategy_ablation.ipynb`).

| Sample | n | Lift without strategy scores | Lift with strategy scores | Strategy-score gain |
|---|---|---|---|---|
| Full scored population | 2,753 | 0.2052 | 0.2716 | 0.0664 |
| Length-matched (within scored population) | 1,235 | 0.0629 | 0.1186 | 0.0557 |

This gain is plotted in Figure 3 (§2.7), alongside the retention percentage discussed below.

Both arms' overall lift collapses sharply once length is matched (69% and 56% reductions,
respectively), consistent with the pattern already established in §3.1.4: length is still doing
most of the work either way. But the strategy-score *gain itself* — the increment the eight scores
add on top of the engineered baseline, holding everything else fixed — drops far less steeply, from
0.0664 to 0.0557, an **84% retention**. This is a markedly smaller erosion than `word_count`'s own
lift or the engineered pipeline's own lift, both of which retained only about 18% of their
full-dataset values under the (differently constructed) strict match in §3.1.4. As with every
length-matching result in this report, this indicates that the strategy-score signal is not
primarily a proxy for comment length under the specific matching procedure applied; it does not by
itself establish complete independence from length or any causal claim about the strategies'
effect on persuasion.

### 3.2.4 Compact models

The nine-column combination and the four-column robust core reported below were identified by
different routes, described fully in §2.7–§2.8: the nine-column combination by an exploratory
search evaluated directly against this population's own held-out test partition, the robust core by
a fixed Mann-Whitney significance criterion rather than a search. Both qualifications carry through
to how the figures below should be read.

Building up from the strategy scores alone, family by family (mirroring the engineered-feature
ablation methodology of §3.1), identified a configuration smaller than the full 237-column set that
exceeds it: the eight strategy scores together with `word_count` alone, nine columns in total,
reached order-robust PR-AUC **0.5379 ± 0.0109**, ahead of the full 237-column combination (0.5098 ±
0.0200). A second, independent estimate of this same nine-column configuration, computed in the
course of the pruning check described below, gave a consistent 0.5380 ± 0.0108 on the identical test
fold — a noise-level difference between the two estimates, itself illustrating the column-order
spread discussed throughout this section. (A four-column subset of this same model, reported below,
reaches a nominally higher PR-AUC still, but not distinguishably so; the two should be read
together, not as an unqualified ranking.) Re-running the classifier/resampling grid restricted to
the nine columns found Random Forest with oversampling
scoring comparably to the originally selected XGBoost configuration (0.5432 ± 0.0065 versus 0.5380 ±
0.0108) — a difference smaller than the combined order-noise band of the two architectures compared,
and not a basis for preferring one over the other on this feature set; XGBoost was retained for
consistency with the rest of the project.

The robust core — refutation, justification, simplification, and `word_count` (selection criterion
in §2.8) — reached order-robust PR-AUC **0.5489 ± 0.0046**. Every tested removal of the other five
strategies moved PR-AUC by at most −0.016, smaller than the combined order-noise band of the
configurations being compared (the sum of their two respective SDs), so none of these differences
is itself a confirmed effect. This is a
higher point estimate than either of the nine-column estimates above, and in that sense the highest
raw PR-AUC among the configurations evaluated against this population, but not distinguishably
better than the nine-column model once the two configurations' combined order-noise band (0.0046 +
0.0108 = 0.0154, against a gap of only 0.0109) is taken into account. The honest reading is that
question-asking, personal narrative, and evidence can be dropped without a measurable cost to
predictive power, not that the four-column model is confirmed superior. The robust core was fixed
in advance of the external validation reported in §3.3, not selected by searching for the
best-performing subset on that external dataset.

### 3.2.5 Individual strategy importance: SHAP versus Mann-Whitney

Two complementary interpretive analyses were applied to the nine-column model (eight strategy
scores plus `word_count`): a SHAP (SHapley Additive exPlanations) analysis using a tree explainer on
the fitted XGBoost model, which describes how much the *fitted model* relies on each feature, and a
model-free Mann-Whitney U comparison converted to a rank-biserial correlation effect size per
strategy, with Benjamini-Hochberg false-discovery-rate correction across the eight simultaneous
tests. Both were computed on the full 2,753-comment scored population and on the strict
length-matched sample from §3.2.3. These two measures answer different questions — model reliance
versus a model-independent univariate association — and neither is a causal-effect estimate.

**Table N.** SHAP importance and Mann-Whitney rank-biserial effect size, full population versus
length-matched sample, for all eight strategies (`word_count`'s own SHAP value — 1.0245 full
population, 0.8402 matched — is omitted here as it dwarfs every strategy's and is not itself one of
the eight strategies under comparison). `*` marks FDR < 0.05 significance
(`07_strategy_interpretation.ipynb`).

| Strategy | SHAP, full population | SHAP, length-matched | Rank-biserial r, full population | Rank-biserial r, length-matched |
|---|---|---|---|---|
| Justification | 0.4156 | 0.1956 | 0.3584* | 0.1309* |
| Concession | 0.3439 | 0.3445 | 0.1481* | 0.0206 |
| Evidence | 0.3592 | 0.3090 | 0.1018* | −0.0322 |
| Refutation | 0.6034 | 0.5193 | 0.3149* | 0.1540* |
| Personal narrative | 0.1853 | 0.2168 | −0.0217 | −0.0859* |
| Call-to-action | 0.1837 | 0.2707 | 0.1584* | 0.0592 |
| Simplification | 0.3193 | 0.3383 | 0.1944* | 0.0789* |
| Question-asking | 0.1518 | 0.1806 | 0.0069 | −0.0345 |

The SHAP columns of this table are plotted in Figure 5 (§2.8).

On the full population, refutation has the highest SHAP importance among the eight strategies
(0.6034, roughly 1.5× the next-highest), while justification has the largest rank-biserial effect
size (0.3584), ahead of refutation's (0.3149). SHAP and the univariate comparison disagree about
which single strategy is most important even before length is controlled for; this is a genuine
point of disagreement between two legitimately different measures, not resolved by preferring one
over the other.

### 3.2.6 Refutation, justification, and the effect of length matching

Refutation and justification diverge sharply once length is controlled for. Refutation's SHAP
importance declines only modestly (0.6034 to 0.5193, holding the top rank among strategies) and its
rank-biserial effect size retains 49% of its full-population value (0.3149 to 0.1540, remaining
significant). Justification's SHAP importance collapses from the model's third most influential
input (`word_count` and refutation ranking first and second) to one of its least influential
strategies (0.4156 to 0.1956) — most of what the fitted model
was using justification for on the full population appears length-related — while its rank-biserial
effect size retains only 37% of its value (0.3584 to 0.1309) but *remains* statistically significant
after FDR correction. Simplification is a modest, consistent third strategy across both measures
(SHAP essentially unchanged, 0.3193 to 0.3383; rank-biserial retaining 41%, 0.1944 to 0.0789).

One further result is worth flagging explicitly rather than leaving implicit: personal narrative's
rank-biserial effect is not significant on the full population but *is* significant, and negative,
on the length-matched sample (r = −0.0859) — the only strategy whose significance status and sign
both shift between the two populations. Concession, call-to-action, and evidence all lose
significance entirely once length is matched, consistent with their apparent full-population effect
being substantially a length artifact.

These results narrow §3.2.2–§3.2.3's group-level finding — that strategy scores as a block carry
signal beyond length — to a more specific pattern: on this dataset, that signal concentrates most
clearly in refutation, with justification and simplification contributing more modestly and
unevenly depending on which measure is consulted, and the remaining four strategies contributing
little that survives length control on their own. As noted in §3.2.1, refutation and justification
are also the two strategies with the weakest human-judge agreement in the scoring validation — a
caveat that bears directly on how strongly this specific pattern should be read, not a separate,
unrelated limitation.

### 3.2.7 Scoring reliability and column-order sensitivity: qualifications

Two distinct sources of measurement uncertainty apply to every result in this section and should
not be conflated. First, the LLM-assigned strategy scores themselves carry the reliability profile
reported in §3.2.1 — strong self-consistency, weaker and more variable agreement with a single
human judge on a small (n = 18) sample, with refutation and justification specifically flagged as
the hardest strategies to calibrate. Second, the fitted model's PR-AUC carries the column-order
sensitivity already described (§2.3, §3.2.2) — a different and independent source of uncertainty
that does not reflect sampling error. Both sources bear on how confidently the refutation/
justification pattern in §3.2.6 should be stated.

Whether that specific pattern — refutation as the most length-robust strategy, justification as the
most length-confounded one — reflects something general about rhetorical persuasion on CMV, or
instead reflects a property of how this dataset's own non-convincing examples happen to be
selected, is the question §3.3 turns to next, using an independently constructed corpus built with
a different negative-example methodology.

## 3.3 RQ3: Cross-Corpus Evaluation on Winning Arguments

### 3.3.1 The WA evaluation population and evaluation tasks

The same eight-strategy rubric, scoring prompt, and fixed model architecture and training
configuration already selected on the home dataset (`RobustScaler`, SMOTE oversampling, XGBoost)
were applied to Winning Arguments (WA; Tan et al., 2016), an independently constructed
r/changemyview corpus whose non-winning replies are deliberately selected to be topically similar to
the corresponding winning reply, in contrast to the home dataset's unfiltered leaf-comment negatives
(§2.1, §2.9). A total of 8,106 direct top-level replies across 3,051 original posts were collected
and scored; 10 comments were dropped after an unparseable LLM response, leaving **8,096 scored
comments**, evaluated on WA's own native training (6,584 comments) and held-out (1,512 comments)
partitions — the same split the corpus is distributed with and on which its own published benchmark
figures are computed. The held-out fold's own positive rate is 52.9%, giving a no-skill PR-AUC
baseline of **0.5291** — far higher than the home dataset's no-skill baselines (roughly 0.20–0.24
depending on the fold), reflecting WA's near-balanced, pairwise construction; raw PR-AUC values are
not comparable between the two datasets without reference to each one's own no-skill baseline.

The home-selected architecture and training configuration were reused on WA, but refit on WA's own
native training split and evaluated on its native held-out split — a transfer of the modeling
procedure, not a zero-shot evaluation of the model already fitted on the home dataset (full protocol
in §2.9). Because WA was scored only on the eight-strategy rubric and `word_count`, the feature set
available here is limited to combinations of the nine columns identified on the home dataset
(§3.2.4), not the larger 237-column representation.

### 3.3.2 Predictive performance: word count, strategy scores, and the robust core

**Table N.** Order-robust PR-AUC and PR-AUC lift over WA's own no-skill baseline (0.5291), for four
feature-set configurations evaluated on WA's native held-out split
(`08_wa_external_validation.ipynb`).

| Feature set | PR-AUC (mean ± SD) | Lift over no-skill |
|---|---|---|
| `word_count` only | 0.5767 ± 0.0000 | 0.0476 |
| Strategy scores only | 0.5967 ± 0.0028 | 0.0676 |
| Strategy scores + `word_count` | 0.5814 ± 0.0058 | 0.0523 |
| Robust core (refutation, justification, simplification, `word_count`) | 0.5919 ± 0.0026 | 0.0628 |

[FIGURE: `figures/report/figure7_wa_external_validation.png` — **Figure N.** Held-out PR-AUC and
pairwise accuracy for feature-set configurations evaluated on Winning Arguments, with literature
benchmark ranges shown for context (`08_wa_external_validation.ipynb`).]

Every strategy-containing configuration beats `word_count` alone by PR-AUC. Unlike on the home
dataset, strategy scores alone (0.5967) slightly edge out strategy scores combined with `word_count`
(0.5814) on this metric — `word_count`'s own correlation with the outcome is markedly weaker on WA
(r = 0.1516) than on the home dataset (r = 0.3716), so combining it with the strategy scores has
less to add here and, by this particular metric, even costs a little instead of adding.

A length-matched robustness check specific to WA — built with the same decile-matching procedure
used in §3.2.3, applied to WA's own (considerably weaker) length confound — found `word_count`'s
own correlation with the outcome dropping from r = 0.1516 to r = 0.1168 (a 23% reduction, much
smaller than the home dataset's 79% reduction under the analogous match, because WA's winners and
losers are already fairly similar in length before any matching). Comparing `word_count` alone
against strategy scores plus `word_count` on this matched sample (7,710 comments), `word_count`
alone lost more than half its lift (0.0476 to 0.0211, 44% retention) while the strategy-inclusive
combination retained most of its lift (0.0523 to 0.0455, 87% retention) — the same qualitative
pattern observed on the home dataset in §3.2.3. This check is narrower than the §3.2.2–§3.2.3
ablation on the home dataset, however: it compares `word_count` alone against `word_count` plus
strategy scores, not an engineered-feature pipeline against that same pipeline plus strategy
scores, since WA was never scored on the home dataset's engineered-feature pipeline. It should be
read as supporting evidence that the strategy signal is not solely a length proxy on WA as well, not
as a replication of the home dataset's specific engineered-pipeline-based ablation.

### 3.3.3 Pairwise accuracy

Because WA's winner/loser pairs are the unit the published literature on this corpus reports
accuracy against, pairwise accuracy — the fraction of held-out pairs for which the model scores the
actual winning reply higher than the actual losing reply — was also computed, over the 805 of 807
native held-out pairs with both sides successfully scored (2 pairs lost both sides to the
unparseable-response drop).

**Table N.** Pairwise accuracy on WA's 805 scored held-out pairs, with published literature
reference ranges for context (Labruna et al., 2026, Table 1).

| Feature set | Pairwise accuracy |
|---|---|
| `word_count` only | 55.59% |
| Strategy scores only | 57.27% |
| Strategy scores + `word_count` | 58.09% |
| Robust core (refutation, justification, simplification, `word_count`) | 58.68% |
| Independent Scoring (literature) | 53.78%–56.66% |
| Multi-Strategy Persuasion Scoring (literature) | 60.59%–64.53% |

The robust core's 58.68% sits above the literature's "Independent Scoring" baselines and below its
full multi-strategy method. This is a reasonable, honestly bounded result, not a like-for-like
comparison: the robust core uses none of this project's engineered features, no tuning against WA,
and a different (and much simpler) scoring and modeling pipeline than the literature's own method.

### 3.3.4 Cross-corpus changes in strategy rankings

**Table N.** SHAP importance and Mann-Whitney rank-biserial effect size for all eight strategies,
home dataset versus WA, each on its own full population (`07_strategy_interpretation.ipynb`;
`08_wa_external_validation.ipynb`). The two datasets' models are fit independently; SHAP magnitudes
are not compared across datasets, only each dataset's own internal ranking.

| Strategy | SHAP rank, home | SHAP rank, WA | Rank-biserial r, home | Rank-biserial r, WA |
|---|---|---|---|---|
| Refutation | 1st (0.6034) | 7th (0.0904) | 0.3149 (2nd) | 0.0949 (4th) |
| Justification | 2nd (0.4156) | 1st (0.2798) | 0.3584 (1st) | 0.1962 (1st) |
| Evidence | 3rd (0.3592) | 2nd (0.2136) | 0.1018 (6th) | 0.1467 (2nd) |
| Simplification | 5th (0.3193) | 3rd (0.1554) | 0.1944 (3rd) | 0.1105 (3rd) |

(Ranks are each strategy's position among the eight strategies only, on its own dataset and
measure — `word_count`'s own SHAP value, excluded from this ranking for the reasons given in
§3.2.5, is 1.0245 at home and 0.2780 on WA, ahead of every strategy in both cases.)

[FIGURE: `figures/report/figure6_strategy_ranking_cmv_vs_wa.png` — **Figure N.** Rank-biserial
effect size for each strategy, home dataset versus Winning Arguments, computed identically on both
(`07_strategy_interpretation.ipynb`; `08_wa_external_validation.ipynb`).]

[FIGURE: `figures/report/figure9_strategy_ranking_shap_cmv_vs_wa.png` — **Figure N.** Mean |SHAP|
*rank* (not magnitude) for each strategy, home dataset versus Winning Arguments
(`07_strategy_interpretation.ipynb`; `08_wa_external_validation.ipynb`).]

Refutation, the home dataset's standout strategy by SHAP, is near the bottom of WA's ranking by the
same measure (7th of 8) and 4th by rank-biserial effect size. Justification, which collapsed under
length control on the home dataset (§3.2.6), is WA's top strategy by both SHAP and rank-biserial
effect size, and — unlike on the home dataset — does not collapse under WA's own length-matched
check: its rank-biserial effect size retains 84% of its value (0.1962 to 0.1648) rather than the 37%
retained on the home dataset. A column-order-shuffle check on the WA SHAP computation (eight
repeats, mirroring the home-dataset procedure) confirms refutation's low standing is stable rather
than an ordering artifact: it never ranks higher than 6th of 8 strategies, in either WA's full
population or its length-matched sample, across all eight repeats.

A natural question is whether this reversal reflects refutation being scored less reliably on WA. A
dedicated test-retest check (n = 200, matching §3.2.1's design) found the opposite: refutation is
*more* internally consistent on WA (76.5% exact match) than on the home dataset (63.8%, the weakest
of the eight strategies there), and every strategy scores more consistently on WA than on the home
dataset. The ranking reversal does not appear to be a scoring-reliability artifact.

### 3.3.5 Pre-specified features versus configurations explored on WA

The four-feature robust core (refutation, justification, simplification, `word_count`) was fixed in
advance of this external evaluation — though not before the home-corpus statistical analysis that
identified it (§3.2.4, §3.2.5) — by a criterion established there: the three strategies whose
Mann-Whitney effect size remained statistically significant both before and after length matching,
not a search for the best-scoring feature subset. It is not the single best-performing configuration found on WA: a narrower, eight-column
configuration dropping only call-to-action (identified by searching WA's own pruning results, not by
any criterion fixed in advance) reached a slightly higher PR-AUC (0.5937 versus the robust core's
0.5919) and pairwise accuracy (59.10% versus 58.68%). This gap is comparable in size to this
pipeline's own order-induced noise band. The robust core remains the reported external-validation
result specifically because it was specified before WA was examined, not selected by scanning WA's
own results for the best-performing subset; the "−call_to_action" configuration is reported here for
completeness, as a configuration explored on WA, not as a replacement for the pre-specified core.

Taken as a whole, this external evaluation supports two structural conclusions that replicate
cleanly on an independently constructed dataset: rhetorical-strategy scores carry predictive signal
beyond `word_count`, and a small, pre-specified strategy-based feature set is competitive with, or
exceeds, every larger combination tried. The specific claim that refutation is the leading strategy,
and that justification is specifically the one whose importance is length-confounded, does not
replicate — on WA, the two strategies' roles are substantially reversed. A difference between the
two datasets' negative-example construction (§2.9) is a plausible candidate explanation for this
reversal, but no analysis in this project tests it directly; it is reported here as an open question
and examined further, as an explicitly labeled hypothesis, in the Discussion chapter.
