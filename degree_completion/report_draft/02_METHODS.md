# 2. Methods

This chapter describes how the home dataset was constructed, how predictive features were
engineered, and how models were trained and evaluated. It then describes how the project's later
analyses were carried out: the baseline investigation that motivated a change in research
direction, the length-controlled analysis, the rhetorical-strategy scoring and ablation, the
strategy interpretation, and the external validation on an independent corpus. Numerical detail
is included where it is needed to define a dataset, procedure, or evaluation setting; the
corresponding findings and their interpretation are reserved for the Results and Discussion
chapters.

## 2.1 Data Collection and Annotation

Data collection targeted the top 500 all-time posts on r/changemyview, using the Reddit API via
PRAW, with the collection approach adapted from the code released by Priniski and Horne (2018) for
their own study of the same subreddit. After collection and loading, 425 of the 500 requested
posts yielded usable comment-level data, producing 3,974 raw comment rows. Each row corresponds to
one reply (comment), not one post or one thread, since the per-comment delta award is the
project's unit of prediction.

Three columns anchor each row: the original post's text (`original_post`), the preceding context
of the same commenter's own prior turns in the thread (`thread_text`), and the comment's own
cleaned text (`final_comment`). The definition of `thread_text` is an explicit design choice: it
includes only the same commenter's own earlier turns in the discussion, not the full thread
including other participants' replies. This keeps the contextual text consistent in authorial
style and sentiment with the comment being scored, at the acknowledged cost of omitting
same-side arguments that other commenters may have contributed to the same thread.

A documented data-cleaning pass removed three rows: a delta-awarded comment whose body had been
edited post hoc into placeholder text, an automated moderator notice that had been parsed as if it
were a participant comment, and a comment whose body consisted of a numeric grid that interfered
with a readability measure computed later in the pipeline. This left a final dataset of 3,971
comments, of which 641 were delta-awarded (16.1%) and 3,330 were not (83.9%).

The target label, `is_convincing`, is derived directly from whether a comment received a delta
from the original poster. Non-convincing (negative) examples are simply the other replies already
present in the same threads; no additional filtering or matching was applied to select which
replies would serve as negative examples. This is a deliberate point of contrast with the external
validation corpus described in §2.9, whose negative examples are instead selected to be topically
similar to the corresponding positive example. One structural property of the resulting dataset
is worth noting here because it bears on later feature-engineering choices: thread size is
correlated with a thread's apparent convincing rate, since a thread typically yields at most a
small, roughly fixed number of deltas while the number of non-convincing replies scales with how
much the thread branches. This is relevant to any analysis that aggregates at the level of the
original post, not the individual comment, and is revisited where relevant in §2.2.

[FIGURE: `figures/report/figure8_pipeline_diagram.png` — **Figure N.** The project's pipeline as
four broad phases — data preparation, modeling and the length-confound discovery, rhetorical-
strategy analysis, and external validation — each labeled with its governing notebook(s). The
last three phases map one-to-one onto the project's three research questions (RQ1–RQ3, tagged on
the diagram); data preparation is shared setup, not itself one of the three. The full
nine-subsection breakdown is §2.1–§2.9 below.]

## 2.2 Feature Engineering

Feature engineering proceeded in three stages, each implemented in its own notebook: a first pass
building linguistic and stylistic features from the comment and the original post; a second pass
adding large-language-model-derived features and document embeddings; and a third pass constructing
features that relate each comment to its original post directly, instead of describing either side
in isolation.

Several feature families went through a documented sequence of trial, diagnosis, and replacement,
which is reported here because each rejection reflects a specific, diagnosed shortcoming rather
than an unexplained substitution. Sentiment was first estimated with TextBlob, which proved
unsuited to comments of the length typical in this dataset, and then with VADER, which was found to
conflate a comment's negative *topic* (for example, discussing a negative event) with a negative
*tone* toward the original poster's argument; both were superseded by a tone-focused sentiment
score assigned by an LLM (Gemini). Tone was first estimated with a RoBERTa-based emotion
classifier, which collapsed roughly two-thirds of comments into a single "neutral" category and
was judged uninformative as a result; it was replaced by an LLM-assigned categorical tone label
with six categories (logical, passionate, humorous, sarcastic, neutral, other), generated for both
the comment and the original post. Stylistic register was first estimated with a zero-shot
`bart-large-mnli` classifier, which produced visibly incorrect labels on manual spot checks, and
was replaced with an LLM-assigned formal/informal label. Rhetorical appeals — ethos, pathos, and
logos — were first estimated using the Empath lexicon, which was found to track topic instead of
rhetorical structure, and were replaced with LLM-assigned scores on a continuous 0–1 scale for each
appeal; of these, the logos score showed the widest separation between convincing and
non-convincing comments of any feature introduced at this stage.

Several other features were retained without this trial-and-replacement sequence: an
adjective-to-adverb ratio computed with spaCy part-of-speech tagging; two readability measures
(Flesch–Kincaid grade level and reading ease) computed with the `textstat` library; a count of
evidence markers, using a keyword-stem method adapted directly from Priniski and Horne's (2018)
own code, which constitutes a direct instance of code reuse, not an independent
re-implementation; the output of a HuggingFace persuasive-language detector
(`chreh/persuasive_language_detector`); and document embeddings from the sentence-embedding model
`BAAI/bge-base-en-v1.5` (native dimensionality 768), reduced via principal component analysis to
187 components, the smallest number retaining at least 85% of cumulative explained variance. All
infrastructure for calling the LLM — client configuration, retry and timeout handling, checkpointed
batch scoring, and prompt templates — is centralized in a single shared module, reused unchanged
across the feature-engineering, rhetorical-scoring, and external-validation notebooks; this
centralization is itself a modest but genuine contribution to the reproducibility of the pipeline.

The relational-feature notebook computes, for each numeric or categorical comment-side feature
that has a natural original-post-side counterpart, both the original-post value and a derived
"difference" (for numeric features) or "match" (a boolean agreement indicator, for categorical or
near-binary features) quantity. The resulting representation therefore describes not only
properties of the reply in isolation, but also how the reply's measured properties relate to those
of the post it is replying to.

Combining these elements yields 41 handcrafted features: 22 numeric features spanning
comment-side, original-post-side, and difference variants of sentiment, rhetorical-appeal, and
readability measures together with evidence-marker counts; 6 boolean comment–post match
indicators; 3 additional scalar features (two formality indicators and the persuasive-language
detector output); and 10 one-hot-encoded columns produced by encoding the two six-category tone
labels (comment-side and original-post-side), with the first category of each dropped as the
implicit reference level — standard practice to avoid the linear dependency a full one-hot
encoding would introduce, since a row's remaining encoded columns already determine membership in
the dropped category when all of them are zero. Together
with the 187 embedding components, this constitutes a 228-feature "trusted" representation.

A further six features — a lexicon-based sentiment score, the earlier (rejected-in-favor-of-LLM)
categorical tone and style labels, and the earlier-generation ethos, pathos, and logos scores —
were formally flagged as **untrusted**, reflecting the diagnosed shortcomings described above, and
recorded as such in a dedicated file. The rationale for
retaining instead of discarding them was to allow their marginal predictive value to be tested
empirically during model selection (§2.3), instead of excluding them by assumption. Encoding these six
raw features (two of which are categorical and together expand into more columns than one each)
adds 11 columns, yielding a 239-feature "trusted+untrusted" representation. An "embeddings-only"
representation (the 187 components alone, with no handcrafted features) was also constructed, to
assess how much predictive value the handcrafted features add over embeddings alone. All three
representations were carried forward as candidate feature sets into model selection.

**Table N.** Feature-family summary, with each family's source notebook and whether it belongs to
the trusted or untrusted set. For the three one-hot-encoded rows, the category count in the
feature-group name is the true number of labels the classifier assigns; "Columns" is one less per
side, since the first category of each encoding is dropped as an implicit reference level (§2.2).

| Feature / feature group | Variants included | Source notebook(s) | Columns | Set |
|---|---|---|---|---|
| Sentiment (LLM, tone-focused) | comment, OP, diff | 03.1, 03.3 | 3 | Trusted |
| Sentiment (VADER, lexicon-based) | comment only | 03.1 | 1 | Untrusted |
| Tone (LLM, 6 categories, one-hot) | comment, OP | 03.1, 03.3 | 10 (2 sides × 5, first category dropped per side) | Trusted |
| Tone (RoBERTa emotion classifier, 7 categories, one-hot) | comment only | 03.1 | 6 (1 side × 6, first category dropped) | Untrusted |
| Style / formality (LLM-assigned, binary) | comment, OP | 03.1, 03.3 | 2 | Trusted |
| Style (BART-large-MNLI zero-shot, 2 categories, one-hot) | comment only | 03.1 | 1 (1 side × 1, first category dropped) | Untrusted |
| Rhetorical appeals: ethos, pathos, logos (LLM, 0–1 scale) | comment, OP, diff | 03.2, 03.3 | 9 | Trusted |
| Rhetorical appeals: ethos, pathos, logos (Empath lexicon) | comment only | 03.2 | 3 | Untrusted |
| Adjective–adverb ratio | comment, OP, diff | 03.2, 03.3 | 3 | Trusted |
| Readability: Flesch–Kincaid grade level | comment, OP, diff | 03.2, 03.3 | 3 | Trusted |
| Readability: Flesch reading ease | comment only (no OP/diff counterpart — dropped as redundant with `fk_grade_level_op`, r = −0.905) | 03.2 | 1 | Trusted |
| Evidence-marker count | comment, OP, diff | 03.2, 03.3 | 3 | Trusted |
| Comment↔OP match indicators (sentiment, tone, style, ethos, pathos, logos) | boolean agreement, computed from original comment/OP values, not derived from the "diff" columns above (bucketed same/different, not a thresholded difference) | 03.3 | 6 | Trusted |
| Persuasive-language detector output | comment only | 03.2 | 1 | Trusted |
| Document embeddings (`BAAI/bge-base-en-v1.5`, PCA) | comment only | 03.2 | 187 | Trusted |
| **Handcrafted Trusted Total** | | | **41** | Trusted |
| **Handcrafted Untrusted Total** | | | **11** | Untrusted |
| **Trusted (handcrafted + embeddings)** | | | **228** | — |
| **Trusted + untrusted** | | | **239** | — |

## 2.3 Model Training and Evaluation Methodology

The prediction task is binary classification: given the representation of a comment (and,
depending on feature-set variant, its relation to the original post), predict whether it was
delta-awarded. Evaluation uses a thread-grouped 80/20 split (`GroupShuffleSplit`, grouped by
discussion thread), assigning 3,262 comments from 320 threads to
training and 709 comments from 81 threads to a held-out test set, with zero thread overlap between
the two, confirmed programmatically. Grouping the split by thread matters because comments from the
same thread share context — the same original post, often overlapping argumentative content, and
sometimes near-duplicate phrasing across replies — so a standard random split risks leaking
thread-specific information between training and test examples, inflating apparent performance in
a way that would not generalize to genuinely unseen discussions.

Model and feature-set selection used a further, inner split carved from the training data using
the same thread-grouping logic (2,629 comments from 256 threads for inner training; 633 comments
from 64 threads for validation), so that the 709-comment test set was not consulted during
selection. A grid of 27 combinations was evaluated on this inner validation split: three
classifiers (logistic regression, random forest, and XGBoost), three resampling conditions applied
to the inner-training partition (no resampling, random undersampling, and SMOTE oversampling), and
the three feature-set variants from §2.2 (trusted, trusted+untrusted, embeddings-only). SMOTE and
random undersampling were included because the positive class is a minority (roughly 16% of the
full dataset); resampling strategies are a standard response to this kind of imbalance, and
including "no resampling" as a condition allows their marginal value to be assessed empirically.

The primary selection and evaluation metric throughout this project is PR-AUC (average precision),
not accuracy or ROC-AUC. This choice follows directly from the class imbalance: with positive
examples at roughly 16–20% depending on the specific split, a classifier that never predicts the
positive class can already achieve high accuracy, and ROC-AUC can look favorable even when
precision on the minority class is poor, because ROC-AUC is computed against the (large) majority
class's true-negative rate. PR-AUC is sensitive specifically to the precision–recall trade-off on
the minority class and is the more informative metric under this kind of imbalance. The
corresponding no-skill reference for a given evaluation fold is the PR-AUC achieved by a
majority-class ("always predict negative") classifier, which is numerically equal to that fold's
own positive rate; because the train/test split is grouped by thread, not stratified by
label, a given fold's own positive rate can differ somewhat from the full dataset's 16.1% rate (the
709-comment test fold used throughout this report has a positive rate of approximately 20.3%), and
the appropriate no-skill baseline for any PR-AUC reported later in this report is that specific
fold's own rate, not the whole-dataset rate.

Selection on the inner validation split identified XGBoost with SMOTE oversampling, on the
239-feature trusted+untrusted representation, as the best-performing combination. This
configuration was then evaluated once on the held-out 709-comment test set, achieving PR-AUC 0.3848.
This number, and its subsequent revision after the length confound described in §2.4 was
discovered, together motivate the remainder of this chapter.

**Table N.** The full 27-combination inner-validation grid, sorted by PR-AUC. These are
**validation**-set scores, used only for selection — not the 0.3848 **test**-set PR-AUC reported
above for the selected configuration, which this grid does not itself contain (the selected row's
own validation PR-AUC is 0.3063; the test-set figure comes from a separate, one-time evaluation
after selection, described in the paragraph above). ROC-AUC is included as the secondary metric
discussed earlier in this section; PR-AUC alone decided selection.

| Classifier | Resampling | Feature set | Validation PR-AUC | Validation ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | **SMOTE oversampling** | **trusted+untrusted** | **0.3063** | **0.6952** |
| Random Forest | SMOTE oversampling | embeddings-only | 0.3043 | 0.6455 |
| Random Forest | SMOTE oversampling | trusted+untrusted | 0.2961 | 0.7088 |
| Random Forest | Random undersampling | trusted+untrusted | 0.2957 | 0.6933 |
| XGBoost | SMOTE oversampling | embeddings-only | 0.2952 | 0.6551 |
| Random Forest | Random undersampling | trusted | 0.2886 | 0.6989 |
| Random Forest | SMOTE oversampling | trusted | 0.2879 | 0.6999 |
| XGBoost | No resampling | trusted | 0.2845 | 0.6983 |
| XGBoost | No resampling | trusted+untrusted | 0.2816 | 0.7070 |
| Random Forest | Random undersampling | embeddings-only | 0.2673 | 0.6413 |
| XGBoost | Random undersampling | trusted | 0.2628 | 0.6659 |
| Logistic Regression | SMOTE oversampling | trusted+untrusted | 0.2574 | 0.6774 |
| Logistic Regression | SMOTE oversampling | trusted | 0.2530 | 0.6748 |
| XGBoost | Random undersampling | embeddings-only | 0.2485 | 0.6140 |
| XGBoost | Random undersampling | trusted+untrusted | 0.2461 | 0.6308 |
| XGBoost | SMOTE oversampling | trusted | 0.2436 | 0.6411 |
| Random Forest | No resampling | trusted | 0.2420 | 0.6157 |
| Logistic Regression | No resampling | trusted | 0.2353 | 0.6587 |
| Logistic Regression | No resampling | trusted+untrusted | 0.2347 | 0.6589 |
| Random Forest | No resampling | trusted+untrusted | 0.2285 | 0.6082 |
| Logistic Regression | SMOTE oversampling | embeddings-only | 0.2285 | 0.6473 |
| XGBoost | No resampling | embeddings-only | 0.2224 | 0.5963 |
| Logistic Regression | No resampling | embeddings-only | 0.2095 | 0.6302 |
| Logistic Regression | Random undersampling | trusted | 0.1816 | 0.5537 |
| Logistic Regression | Random undersampling | trusted+untrusted | 0.1814 | 0.5541 |
| Logistic Regression | Random undersampling | embeddings-only | 0.1786 | 0.5477 |
| Random Forest | No resampling | embeddings-only | 0.1506 | 0.4468 |

This validation fold's own no-skill PR-AUC baseline is 0.1690 (its own positive rate — not the
whole-dataset rate of 0.1614, which this fold's rate differs from, consistent with how a
thread-grouped split's fold-specific positive rate generally departs somewhat from the
whole-dataset rate elsewhere in this report). All but one of the 27 combinations beat it; the sole
exception is the weakest row, `RF (no resampling) | embeddings-only` (0.1506), which falls below both this
fold's own baseline (0.1690) and the whole-dataset rate (0.1614) — the only combination in the grid
that fails to beat no-skill under either baseline. `RF (oversampling) | embeddings-only` (0.3043) trails
the selected configuration (0.3063) by 0.002 PR-AUC. For `XGB (oversampling)` specifically, PR-AUC on
`trusted+untrusted` exceeds PR-AUC on `trusted` alone by 0.063 — the largest such difference among
the 9 classifier/resampling pairs.

**Column-order sensitivity.** The `XGB (oversampling)` architecture used throughout this chapter is
sensitive to the order its input columns are presented in, not just which columns are included.
XGBoost builds each decision tree by selecting, at every split, whichever candidate feature yields
the greatest reduction in training loss; when two or more features are correlated — common
throughout this project's feature sets, from the 187 embedding components to the engineered
features' own overlapping signals — their split-gain values can come out essentially tied, and
such ties are broken deterministically by each feature's position (column order) in the input
matrix, not by anything about the feature's actual values. This has been found to swing PR-AUC by
as much as 0.0482 on one of this project's larger feature representations (the 237-column set
tested in §2.7), purely from reordering the same columns. The sensitivity is specific to the
tree-building step, not `SMOTE`'s oversampling: `SMOTE`'s neighbor-finding operates on Euclidean
distance, which does not depend on the order features are summed in. Because of this, every PR-AUC
value reported from this point in the chapter onward is computed as a mean and standard deviation
across eight column orderings (the original ordering plus seven random reshufflings) of the same
underlying feature matrix, not as a single model fit. These standard deviations describe
sensitivity to an essentially arbitrary implementation detail — column order — and are not
bootstrap confidence intervals or any other estimate of sampling uncertainty; they should not be
interpreted as such.

## 2.4 Baseline Investigation and the Length Confound

Having selected and evaluated the engineered-feature pipeline, a substantially simpler baseline was
evaluated on the identical 709-comment test fold, as a check on whether the engineered
representation was adding value over a much cheaper alternative: TF-IDF vectorization of the
comment text alone, with a logistic regression classifier and no engineered features at all. This
baseline outperformed the selected 239-feature engineered pipeline on the same test fold, reaching
PR-AUC 0.5119 against the engineered pipeline's 0.3848.

This result prompted a diagnostic step: examining which tokens
the TF-IDF model weighted most heavily revealed that the great majority were common stopwords,
together with the token "edit" (characteristic of Reddit's edit-marker convention) — a pattern more
consistent with the model tracking how much text a comment contains than with it tracking
rhetorical content.

This diagnostic finding motivated testing the role of comment length directly, in two ways. First,
a second, even simpler baseline was evaluated: a single-feature logistic regression using only the
comment's raw word count. This baseline outperformed TF-IDF too, reaching PR-AUC 0.5256 — the
strongest of the three models compared so far, using the least information.

[FIGURE: `figures/report/figure1_initial_model_comparison.png` — **Figure N.** Held-out PR-AUC for
the originally selected engineered-feature pipeline, a TF-IDF baseline, and a word-count-only
baseline, evaluated on the identical 709-comment test fold (`04.2_baseline_comparison.ipynb`).]

Second, the raw relationship between comment length and outcome was characterized directly:
comment length and outcome are strongly associated in this dataset, with a mean word count of 78.0
for non-convincing comments and 216.7 for convincing comments, a point-biserial correlation of
0.3716 (denoted r hereafter). To rule out the possibility that this association was
purely an artifact of some discussion
threads being systematically longer or more active than others (and hence containing both longer
comments and more opportunities for a delta), a within-thread check was performed: restricting to
the 200 threads that contained at least one comment of each outcome class, the convincing comment
was longer than that same thread's own average non-convincing comment in 153 of the 200 threads
(76.5%).

In response to this finding, `word_count` was added to the engineered feature set as an additional
column, and the full 27-combination selection grid from §2.3 was rerun from scratch on the
word-count-augmented feature sets, instead of simply appending the feature to the previously
selected configuration. The same architecture was selected again — XGBoost with SMOTE
oversampling on the trusted+untrusted feature set, now including `word_count` — and this recovered
most of the shortfall relative to the simpler baselines: test PR-AUC rose from 0.3848 to 0.4922.

These two figures are each the mean of multiple executions across eight column orderings, not a
single model fit, per the column-order-sensitivity convention established in §2.3. This gives
0.3676 ± 0.0182 for the originally selected pipeline and 0.4831 ± 0.0257 after the
word-count-augmented rerun, with bootstrap 95% confidence intervals of [0.311, 0.463] and
[0.410, 0.586] respectively. The remainder of
this report treats the recognition and correction of this length confound as a methodological
turning point: it reframes the project's central question from whether textual features predict
delta awards to which predictive signal, if any, remains once comment length is accounted for — the
question addressed in §2.5 onward.

[FIGURE: `figures/report/figure4_comment_length_by_outcome.png` — **Figure N.** Word-count
distribution by outcome for the full 3,971-comment dataset, with each class's mean indicated
(`04.2_baseline_comparison.ipynb`).]

**Table N.** The four models discussed above, all evaluated on the identical 709-comment test
fold (no-skill PR-AUC baseline 0.2031). Rows 1 and 4 use the order-sensitive XGBoost+SMOTE
architecture (§2.7), so both of their metrics are reported as order-robust means ± SD across 8
column-order shuffles. Rows 2–3 (logistic regression) have
no such sensitivity and are reported as single-fit point estimates, which are already their stable
values.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Originally selected engineered pipeline (XGBoost + SMOTE, trusted+untrusted) | 0.3676 ± 0.0182 | 0.6903 ± 0.0096 |
| TF-IDF + Logistic Regression | 0.5119 | 0.8150 |
| Word count alone (1 feature) | 0.5256 | 0.8294 |
| Word-count-augmented re-selected pipeline (XGBoost + SMOTE, trusted+untrusted + `word_count`) | 0.4831 ± 0.0257 | 0.7923 ± 0.0073 |

## 2.5 Length-Controlled Analysis

Folding `word_count` into the feature set and reselecting recovers predictive performance, but does
not on its own establish whether any part of the recovered performance reflects something beyond
length — in principle, a model's entire predictive value could still reduce to "longer comments are
more often convincing." A dedicated length-controlled analysis was carried out on the full
3,971-comment dataset to address this directly, using two different constructions to reduce the
length imbalance between outcome classes: a loose word-count floor (84 words — the convincing
group's own 25th percentile), which discards the shortest non-convincing comments and yields a
sample of 1,638 comments, and a strict decile-bin length match, which bins convincing comments'
`word_count` into ten equal-count groups and samples a matching number of non-convincing comments
from each group, yielding a sample of 1,235 comments. For each of these two samples, and for the full, unmatched dataset, the
lift of several models — TF-IDF, word-count-only, and the full engineered pipeline — over that
sample's own no-skill baseline was computed, allowing each model's apparent skill before and after
length control to be compared directly.

Across all three models tested, apparent predictive lift drops substantially once length is
controlled for — typically by roughly three-quarters to seven-eighths, depending on the specific
model and the strictness of the matching — but in no case falls to zero or below. This pattern
indicates that some predictive signal beyond raw length survives length control for every approach
tested, though the degree of that survival differs considerably between models and is, on its own,
modest in absolute terms. A further, separately noted methodological result from this analysis
concerns the risk of re-selecting a model on a small sample: re-running the complete 27-combination
selection grid from §2.3 from scratch on the small (1,235-row) strictly length-matched sample
selected a different architecture that, when evaluated on the corresponding test partition, scored
*below* its own no-skill baseline — a concrete instance of overfitting attributable to comparing
many candidate configurations (27) against a small validation partition (205 rows). By contrast,
simply reusing the architecture already selected on the full dataset, without re-selecting, scored
positively on the same matched test partition. This result is presented as a specific, worked
illustration of a general methodological risk — model re-selection on small samples — and not as
evidence against re-selection as a practice in general.

**Table N.** The two length-controlled samples used in this section, side by side (both built in
`04.3_length_controlled_analysis.ipynb`, from the full 3,971-comment dataset).

| Sample | Matching procedure | n | `word_count` r (unmatched → matched) |
|---|---|---|---|
| Loose length floor | Keep every convincing comment; restrict non-convincing comments to ≥ the convincing group's 25th-percentile length (84 words) | 1,638 | 0.372 → 0.093 |
| Strict decile-bin match | Decile-bin match: convincing comments' `word_count` split into 10 equal-count bins; non-convincing comments sampled from the matching bin | 1,235 | 0.372 → 0.066 |

## 2.6 Rhetorical-Strategy Scoring

The length-controlled analysis in §2.5 establishes that some non-length signal survives across
several modeling approaches, but it does not characterize what that signal consists of. To
investigate whether an argument's manner of construction carries predictive information beyond
simply how much is written, eight rhetorical strategies relevant to CMV-style argumentation were
defined: justification, concession, evidence, refutation, personal narrative,
call-to-action, simplification, and question-asking. Each strategy is scored on an ordinal 1–5
scale, independently of the others, by an LLM (Gemini). The general approach — using an LLM to
assign numeric persuasion-strategy scores to naturally occurring text — follows the methodology
described by Labruna, Modzelewski, Satta, and Da San Martino (2026), who applied an analogous
scoring approach primarily to the Winning Arguments, Anthropic/Persuasion, and Persuasion for Good
corpora. The eight strategies used in this project were defined independently for the CMV setting
and are not a reuse of Labruna et al.'s own six-category taxonomy, although two strategy names
(justification, simplification) happen to be shared.

Scoring was restricted to comments of at least 30 words, or of any length if the comment was
delta-awarded, yielding a scored population of 2,753 comments across 400 original posts — a subset
of the full 3,971-comment dataset, not the full dataset itself. This restriction reflects a
judgment that very short comments carry too little content for a meaningful rhetorical-strategy
judgment, while the "or delta-awarded" clause ensures that no positive example is discarded
regardless of its length. Because the restriction disproportionately removes short, non-convincing
comments while retaining every convincing comment, the resulting subset's convincing-comment
prevalence (641 of 2,753 comments, 23.3%) is substantially higher than the full dataset's 16.1%.
Later sections that report figures computed within this scored subset (§2.7 onward) use this
subset's own denominator, not the full dataset's 16.1% — the two should not be read as directly
comparable.

The scoring prompt went through one documented revision. An initial version presented each comment
to the LLM without including the original post it was replying to. Review of this initial pass
identified that two of the eight strategies — concession and refutation — are defined relative to
a specific point made in the original post, and cannot be judged accurately without the LLM having
access to that point. A revised prompt batches comments by original post and includes the post's
own text as context. In a manually reviewed comparison sample, refutation scores increased for the
majority of comments under the revised prompt (6 of 14 reviewed rows increased, 1 decreased),
consistent with the earlier, original-post-blind prompt having systematically under-detected
refutation. The full scoring run used the revised, original-post-aware prompt.

Two reliability checks were carried out, both explicitly framed as lightweight checks, not a
formal inter-annotator-agreement study. The first is a test-retest check: 200
comments were scored twice, independently, and the two passes were compared. Agreement was strong
across all eight strategies, with 95–99% of repeat scores falling within one point of the original
score (mean absolute difference between 0.14 and 0.41 depending on strategy). The second is a
comparison against a single human judge's hand-scoring of 18 comments (9 convincing, 9 not). These
figures are the judge's final scores, after she reviewed her own first-pass disagreements with the
LLM and corrected some of them; initial agreement was considerably weaker (exact-match as low as
11.1%, for justification, before this review). At the same ±1-point tolerance used for the
test-retest check above, final agreement reaches at least 83.3% for every strategy, with mean
absolute differences between 0.28 and 0.89 — weakest specifically for refutation and justification,
the two hardest strategies to calibrate against human judgment even after review. This reliability
pattern is directly relevant to results reported later in this project, since refutation and
justification also prove to be the two strategies most central to the project's predictive and
interpretive findings (§2.8).

**Table N.** The eight rhetorical strategies, with the test-retest (n = 200) and human-agreement
(n = 18, the judge's final scores after reviewing her own first-pass disagreements with the LLM —
see prose above) reliability figures for each, both at the same ±1-point tolerance.

| Strategy | Definition | Test-retest: within ±1 pt | Test-retest: mean abs. diff. | Human agreement: within ±1 pt | Human agreement: mean abs. diff. |
|---|---|---|---|---|---|
| Justification | Gives a reason or logical basis for the position, beyond simply asserting it | 98.5% | 0.29 | 88.9% | 0.89 |
| Concession | Acknowledges a valid point, limitation, or partial truth in the opposing view before responding | 99.0% | 0.29 | 94.4% | 0.67 |
| Evidence | Cites a concrete fact, statistic, study, source, or specific example to support a claim | 99.5% | 0.24 | 88.9% | 0.72 |
| Refutation | Directly identifies and counters a specific flaw, assumption, or weak point in the original argument | 95.5% | 0.41 | 83.3% | 0.89 |
| Personal narrative | Draws on the commenter's own lived experience to make the case | 99.5% | 0.14 | 94.4% | 0.28 |
| Call-to-action | Offers a concrete alternative way to act on or think about the issue, not just criticism | 95.5% | 0.30 | 100.0% | 0.67 |
| Simplification | Uses an analogy, thought experiment, or simplified restatement to make an abstract point concrete | 98.0% | 0.27 | 100.0% | 0.28 |
| Question-asking | Poses a genuine question back to the original poster that probes an assumption or invites clarification | 98.0% | 0.16 | 94.4% | 0.28 |

## 2.7 Strategy Ablation

To test whether the eight rhetorical-strategy scores add predictive information beyond an existing
engineered-feature representation, the scores were appended to an engineered-feature baseline and
the combined representation was evaluated using the same thread-grouped evaluation logic used
elsewhere in this project, restricted to the 2,753-comment rhetorical-scoring population from
§2.6. The engineered-feature baseline used in this analysis is the 228-feature trusted
representation from §2.2 with `word_count` added (229 columns in total); it does not include the
eleven untrusted-derived columns that formed part of the 239- and 240-feature representations
selected and evaluated in §2.3 and §2.4. This is noted explicitly because it means the ablation
described in this section is not a direct continuation of the exact feature set carried through
§2.3–§2.4, but a related, narrower representation built specifically for this analysis. Combined
with the eight strategy
scores, the resulting representation totals 237 columns.

The engineered-feature baseline alone achieves PR-AUC 0.4434 (± 0.0187); adding the
eight strategy scores raises this to 0.5098 (± 0.0200), a gain of 0.0664. The eight strategy scores
alone, with no engineered features at all, achieve PR-AUC 0.3514 (± 0.0065) — a real, standalone
predictive signal, positioned between this population's own no-skill baseline (0.2382) and the
combined 237-column model.

A further, parsimony-oriented finding is reported here because it bears on how the engineered and
strategy-score features interact. Among all feature combinations evaluated, the smallest — the
eight strategy scores together with `word_count` alone (nine columns in total) — achieved the
highest PR-AUC of any configuration tested in this project, 0.5379 (± 0.0109), exceeding the full
237-column representation. Re-running the classifier and resampling grid restricted to these nine
columns found that a random forest classifier performed comparably to the originally selected
XGBoost configuration (0.5432 ± 0.0065 versus 0.5380 ± 0.0108) — a difference smaller than either
classifier's own order-induced spread, and therefore not a basis for preferring one architecture
over the other on this particular feature set.

This gain's survival under length control is the most direct test of this project's central
claim — that rhetorical-strategy scores carry real signal beyond comment length, not just another
proxy for it — mirroring the same test §2.5 applied to `word_count`'s own contribution. It was
checked using a length-matched sample built specifically for this 2,753-comment population: the
same decile-bin matching procedure used for `04.3`'s strict decile-bin match (§2.5) was independently re-run
here, yielding 1,235 comments — coincidentally the same size
as §2.5's own strict-match sample, despite being built from a different, smaller base population;
the two samples must not be confused or their figures combined. On this length-matched sample, the
strategy-score gain drops from 0.0664 to 0.0557 PR-AUC, an 84% retention — far less erosion than
`word_count`'s own lift or the engineered pipeline's own lift, both of which collapsed to about 18%
of their full-dataset values under `04.3`'s strict matching (§2.5). This indicates the
strategy-score signal is not primarily a proxy for comment length, unlike most other signal
examined in this project.

[FIGURE: `figures/report/figure3_strategy_gain_length_matching.png` — **Figure N.** The
strategy-score PR-AUC gain reported above, recomputed on the length-matched sample within the
2,753-comment scored population, showing how much of the gain survives length control
(`06_strategy_ablation.ipynb`).]

## 2.8 Strategy Interpretation

To identify which individual strategies underlie the aggregate gain reported in §2.7, two
complementary interpretive analyses were applied to the nine-column model (the eight strategy
scores plus `word_count`) identified in that section as the strongest-performing combination. The
first is a SHAP (SHapley Additive exPlanations) analysis, using a tree explainer applied directly to
the fitted XGBoost model, which summarizes how much each feature moves the model's output, on
average, across the comments in a given evaluation fold. The second is a model-free univariate
comparison: for each strategy, a two-sided Mann–Whitney U test compares the distribution of that
strategy's scores between convincing and non-convincing comments, converted to a rank-biserial
correlation as the corresponding effect size; p-values across the eight simultaneous tests are
corrected for multiple comparisons using the Benjamini–Hochberg false-discovery-rate procedure, with
significance assessed at a corrected threshold of 0.05. Both analyses were computed twice: once on
the full 2,753-comment scored population, and once on the strict, decile-bin length-matched sample
described in §2.7, to distinguish which strategies' apparent importance holds up once length is
controlled for from those whose apparent importance is better explained by their correlation with
length.

SHAP values and the Mann–Whitney rank-biserial effect size answer related but distinct questions:
the former describes how much a specific fitted model relies on a feature, while the latter
describes a model-independent, univariate association between a feature's score distribution and
the outcome. Neither is a causal-effect estimate — both describe observed association or model
reliance, not a demonstrated causal mechanism — and the two measures are not guaranteed to agree
with one another, nor are they intended as interchangeable. In this project, these two measures
produce a genuine point of disagreement about which single strategy is most important even on the
home dataset's full population; this is reported directly, as an aspect of the interpretive findings
to be resolved (or left open) in Results and Discussion, not papered over by relying on only
one of the two measures.

**Table N.** Both interpretive measures, full population and length-matched sample, for all eight
strategies (`word_count`'s own SHAP value — 1.0245 full population, 0.8402 matched — is omitted
from this table for the same reason it's excluded from the figure below: it dwarfs every
strategy's and isn't itself one of the eight strategies under comparison). `*` marks FDR < 0.05
significance.

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

One row is worth flagging, not leaving implicit: `personal_narrative`'s rank-biserial effect
is not significant on the full population but *is* significant, and negative, on the length-matched
sample (r = −0.0859) — the only strategy whose significance status and sign both shift between the
two populations. This is reported here as a fact about the data, not interpreted; the interpretive
discussion belongs in Results/Discussion, not Methods.

[FIGURE: `figures/report/figure5_refutation_vs_justification_shap.png` — **Figure N.** Mean |SHAP|
value for each of the eight rhetorical strategies, full population versus length-matched sample,
for the nine-column selected model (`07_strategy_interpretation.ipynb`). A simplified ranked
comparison, not the raw beeswarm plots, per the legibility concern noted above; `word_count` is
excluded from the chart, since the comparison is among the eight strategies themselves, not
against the length confound already established in §2.4.]

## 2.9 External Validation on Winning Arguments

To assess whether the rhetorical-strategy findings reported in §2.7–§2.8 are specific to this
project's own dataset-construction choices, the same scoring rubric and modeling approach were
applied to Winning Arguments (WA; Tan, Niculae, Danescu-Niculescu-Mizil, and Lee, 2016), an
independently constructed persuasion corpus built from paired winning and losing replies on
r/changemyview, accessed through ConvoKit. WA differs from the home dataset in an important
respect: its losing (non-winning) replies are deliberately selected to be topically similar and
substantively comparable to the corresponding winning reply, matched by Jaccard similarity within
the same thread, whereas the home dataset's non-convincing comments are simply the other,
unfiltered replies already present in the same threads (§2.1). This difference in how negative
examples are constructed is a methodologically important point of contrast between the two
datasets, revisited in the Results and Discussion chapters.

A total of 8,106 comments, spanning 3,051 original posts, were collected and scored on the same
eight-strategy rubric, using the same scoring infrastructure described in §2.6. Ten comments were
subsequently dropped following an unparseable response from the LLM, leaving 8,096 scored comments.
These were evaluated on WA's own native training and held-out partitions — the same partition the
corpus is distributed with, and the same partition on which its own published benchmark figures are
computed — comprising 6,584 comments for training and 1,512 for evaluation.

The training and evaluation protocol is stated precisely here because an imprecise description
would misrepresent what was actually done. The model architecture and training configuration
already selected on the home dataset (`RobustScaler`, SMOTE oversampling, and XGBoost) were reused
unchanged on WA, with no new architecture selection, hyperparameter search, or feature-set tuning
performed using WA data. However, a new model instance was then fit on WA's own native training
split and evaluated on WA's own native held-out split. This procedure — reusing a previously
selected modeling configuration, but fitting new model parameters on the new dataset's own training
data — is a transfer of the modeling procedure, not a zero-shot evaluation of the model originally
fitted on the home dataset, and it should not be described as zero-shot transfer or as applying a
frozen model to WA. A separate, exploratory notebook does evaluate genuine zero-shot transfer (the
home-dataset-trained model applied directly to WA with no refitting at all); that analysis,
together with two other exploratory checks in the same notebook, is excluded from the scope of this
report.

Because WA's class balance and negative-sampling procedure differ substantially from the home
dataset's, raw PR-AUC values are not directly comparable between the two datasets without reference
to each dataset's own no-skill baseline; WA's held-out fold has a no-skill PR-AUC of 0.5291,
reflecting its near-balanced, pairwise construction, compared with the considerably lower no-skill
baselines (roughly 0.20–0.24 depending on the specific fold) reported for the home dataset
elsewhere in this chapter. Pairwise accuracy — the proportion of winner/loser pairs for which the
model scores the actual winning reply higher than the actual losing reply — is also reported for
WA, since it is the metric reported in the published literature this analysis is compared against
(Labruna et al., 2026). WA's own pairing procedure yields 4,263 winner/loser pairs overall, with 805
of them falling in the held-out split used to compute this metric.

One limitation of this external validation is stated directly here, because it bears on how the
WA results should be read, not just a matter for the Discussion chapter: this
analysis supports the broader claim that rhetorical-strategy scores carry predictive information in
a second, independently constructed dataset, but it does not, by itself, establish that their
*incremental* predictive contribution specifically beyond comment length replicates to the same
extent documented for the home dataset in §2.7. No length-controlled ablation comparable to the one
described there was performed on WA; establishing that stronger, more specific claim would require
such an analysis, which falls outside the scope of the present external-validation check.

**Table N.** The two datasets' construction, side by side — the negative-example row is the single
most consequential difference between them, discussed above.

| | Home dataset | Winning Arguments (WA) |
|---|---|---|
| Source | r/changemyview, collected directly via the Reddit API (PRAW) | r/changemyview-derived, accessed via ConvoKit (Tan, Niculae, Danescu-Niculescu-Mizil, and Lee, 2016) |
| Unit of construction | Comment-level — every reply in a thread is its own labeled row | Winner/loser pair-level — each original post contributes one matched pair |
| Positive (convincing) examples | Comment awarded a delta | Reply labeled the pair's winner |
| Negative (non-convincing) examples | All other replies already present in the same thread, unfiltered | The single reply most topically similar (Jaccard similarity) to the winner, among replies that did not win |
| Size | 3,971 comments, 425 original posts | 8,096 scored comments (8,106 collected, 10 dropped for unparseable LLM responses), 3,051 original posts |
| Class balance | 641 convincing (16.1%) | Near-balanced by construction (~52% positive — see no-skill baseline below) |
| Train/test split | This project's own thread-grouped 80/20 split: 3,262 train / 709 test | WA's native distributed split (not re-split by this project): 6,584 train / 1,512 heldout |
| No-skill PR-AUC baseline | ≈0.20–0.24, depending on the specific fold | 0.5291 (heldout fold's own positive rate) |
| Paired design | No — comments are evaluated individually, not as pairs | Yes — 4,263 winner/loser pairs total, 805 in the heldout split |
