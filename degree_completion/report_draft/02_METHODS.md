# 2. Methods

This chapter describes how the home dataset was constructed, how predictive features were
engineered, how models were trained and evaluated, and how the project's later analyses — the
baseline investigation that motivated a change in research direction, the length-controlled
analysis, the rhetorical-strategy scoring and ablation, the strategy interpretation, and the
external validation on an independent corpus — were carried out. Numerical detail is included
where it is needed to define a dataset, procedure, or evaluation setting; the corresponding
findings and their interpretation are reserved for the Results and Discussion chapters.

## 2.1 Data Collection and Annotation

Data collection targeted the top-ranked posts on r/changemyview, using the Reddit API via PRAW,
with the collection approach adapted from the code released by Priniski and Horne (2018) for their
own study of the same subreddit. Collection and loading yielded comment-level data from 425 posts,
producing 3,974 raw comment rows. Each row corresponds to one reply (comment), not one post or one
thread, since the per-comment delta award is the project's unit of prediction.

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
original post rather than the individual comment, and is revisited where relevant in §2.2.

[FIGURE PLACEHOLDER: End-to-end research pipeline diagram, showing the sequence data collection →
annotation → feature engineering → model training and selection → baseline investigation →
length-controlled analysis → rhetorical-strategy scoring → strategy ablation and interpretation →
external validation, with the notebook responsible for each stage labeled.]

## 2.2 Feature Engineering

Feature engineering proceeded in three stages, each implemented in its own notebook: a first pass
building linguistic and stylistic features from the comment and the original post; a second pass
adding large-language-model-derived features and document embeddings; and a third pass constructing
features that relate each comment to its original post directly, rather than describing either side
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
logos — were first estimated using the Empath lexicon, which was found to track topic rather than
rhetorical structure, and were replaced with LLM-assigned scores on a continuous 0–1 scale for each
appeal; of these, the logos score showed the widest separation between convincing and
non-convincing comments of any feature introduced at this stage.

Several other features were retained without this trial-and-replacement sequence: an
adjective-to-adverb ratio computed with spaCy part-of-speech tagging; two readability measures
(Flesch–Kincaid grade level and reading ease) computed with the `textstat` library; a count of
evidence markers, using a keyword-stem method adapted directly from Priniski and Horne's (2018)
own code, which constitutes a direct instance of code reuse rather than independent
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
labels (comment-side and original-post-side) with the first category of each dropped. Together
with the 187 embedding components, this constitutes a 228-feature "trusted" representation.

A further six features — a lexicon-based sentiment score, the earlier (rejected-in-favor-of-LLM)
categorical tone and style labels, and the earlier-generation ethos, pathos, and logos scores —
were formally flagged as **untrusted**, reflecting the diagnosed shortcomings described above, and
recorded as such in a dedicated file rather than deleted from the dataset. The rationale for
retaining rather than discarding them was to allow their marginal predictive value to be tested
empirically during model selection (§2.3), rather than excluded by assumption. Encoding these six
raw features (two of which are categorical and together expand into more columns than one each)
adds 11 columns, yielding a 239-feature "trusted+untrusted" representation. An "embeddings-only"
representation (the 187 components alone, with no handcrafted features) was also constructed, to
assess how much predictive value the handcrafted features add over embeddings alone. All three
representations were carried forward as candidate feature sets into model selection.

[TABLE PLACEHOLDER: Feature-family summary, listing each feature or feature group, its source
notebook, its column count, and whether it is part of the trusted or untrusted set.]

[FIGURE PLACEHOLDER: Feature correlation heatmap (handcrafted features, embeddings excluded),
illustrating the redundancy among engineered features that becomes methodologically relevant in
§2.7.]

## 2.3 Model Training and Evaluation Methodology

The prediction task is binary classification: given the representation of a comment (and,
depending on feature-set variant, its relation to the original post), predict whether it was
delta-awarded. Evaluation uses a thread-grouped 80/20 split (`GroupShuffleSplit`, grouped by
discussion thread) rather than a simple random split, assigning 3,262 comments from 320 threads to
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
including "no resampling" as a condition allows their marginal value to be assessed empirically
rather than assumed.

The primary selection and evaluation metric throughout this project is PR-AUC (average precision),
not accuracy or ROC-AUC. This choice follows directly from the class imbalance: with positive
examples at roughly 16–20% depending on the specific split, a classifier that never predicts the
positive class can already achieve high accuracy, and ROC-AUC can look favorable even when
precision on the minority class is poor, because ROC-AUC is computed against the (large) majority
class's true-negative rate. PR-AUC is sensitive specifically to the precision–recall trade-off on
the minority class and is the more informative metric under this kind of imbalance. The
corresponding no-skill reference for a given evaluation fold is the PR-AUC achieved by a
majority-class ("always predict negative") classifier, which is numerically equal to that fold's
own positive rate; because the train/test split is grouped by thread rather than stratified by
label, a given fold's own positive rate can differ somewhat from the full dataset's 16.1% rate (the
709-comment test fold used throughout this report has a positive rate of approximately 20.3%), and
the appropriate no-skill baseline for any PR-AUC reported later in this report is that specific
fold's own rate, not the whole-dataset rate.

Selection on the inner validation split identified XGBoost with SMOTE oversampling, on the
239-feature trusted+untrusted representation, as the best-performing combination. This
configuration was then evaluated once on the held-out 709-comment test set, achieving PR-AUC 0.3848.
This number, and its subsequent revision after the length confound described in §2.4 was
discovered, together motivate the remainder of this chapter.

[TABLE PLACEHOLDER: Model-selection grid summary — three classifiers × three resampling conditions
× three feature-set variants, with inner-validation PR-AUC for each of the 27 combinations and the
selected configuration indicated.]

[FIGURE PLACEHOLDER: Precision–recall curve and ROC curve for the selected pipeline, evaluated on
the held-out test set.]

## 2.4 Baseline Investigation and the Length Confound

Having selected and evaluated the engineered-feature pipeline, two substantially simpler models
were evaluated on the identical 709-comment test fold, as a check on whether the engineered
representation was adding value over much cheaper alternatives. The first used TF-IDF vectorization
of the comment text alone, with a logistic regression classifier and no engineered features at all;
the second used the comment's raw word count as a single feature, again with logistic regression.
Both simpler models outperformed the selected 239-feature engineered pipeline on the same test
fold: the TF-IDF baseline reached PR-AUC 0.5119, and the word-count-only model reached PR-AUC
0.5256, against the engineered pipeline's 0.3848.

This result prompted a diagnostic step rather than an immediate conclusion. Examining which tokens
the TF-IDF model weighted most heavily revealed that the great majority were common stopwords, 
together with the token "edit" (characteristic of Reddit's edit-marker convention) — a pattern more
consistent with the model tracking how much text a comment contains than with it tracking
rhetorical content. This motivated testing the role of comment length directly. Comment length and
outcome are strongly associated in this dataset: the mean word count is 78.0 for non-convincing
comments and 216.7 for convincing comments, a point-biserial correlation of 0.3716. To rule out the
possibility that this association was purely an artifact of some discussion threads being
systematically longer or more active than others (and hence containing both longer comments and
more opportunities for a delta), a within-thread check was performed: restricting to the 200
threads that contained at least one comment of each outcome class, the convincing comment was
longer than that same thread's own average non-convincing comment in 153 of the 200 threads
(76.5%).

In response to this finding, `word_count` was added to the engineered feature set as an additional
column, and the full 27-combination selection grid from §2.3 was rerun from scratch on the
word-count-augmented feature sets, rather than simply appending the feature to the previously
selected configuration. This recovered most of the shortfall relative to the simpler baselines:
test PR-AUC rose from 0.3848 to 0.4922. Reporting these two figures as order-robust means with
standard deviations computed over repeated column-order shufflings of the same underlying features
(the motivation for this convention is described in §2.7) gives 0.3676 ± 0.0182 for the original
selection and 0.4831 ± 0.0257 after the rerun, with bootstrap 95% confidence intervals of
[0.311, 0.463] and [0.410, 0.586] respectively — intervals that barely overlap, making this the
single most statistically decisive comparison produced anywhere in this project. The remainder of
this report treats the recognition and correction of this length confound as a methodological
turning point: it reframes the project's central question from whether textual features predict
delta awards to which predictive signal, if any, remains once comment length is accounted for — the
question addressed in §2.5 onward.

[FIGURE PLACEHOLDER: Word-count distribution by outcome (non-convincing vs. convincing), with each
class's mean indicated.]

[TABLE PLACEHOLDER: Baseline comparison — the originally selected engineered pipeline, the TF-IDF
baseline, the word-count-only baseline, and the word-count-augmented re-selected pipeline, all
evaluated on the identical test fold.]

## 2.5 Length-Controlled Analysis

Folding `word_count` into the feature set and reselecting recovers predictive performance, but does
not on its own establish whether any part of the recovered performance reflects something beyond
length — in principle, a model's entire predictive value could still reduce to "longer comments are
more often convincing." A dedicated length-controlled analysis was carried out on the full
3,971-comment dataset to address this directly, using two different constructions to reduce the
length imbalance between outcome classes: a loose word-count floor, which discards the shortest
non-convincing comments and yields a sample of 1,638 comments, and a strict one-to-one length match,
which pairs each convincing comment with a non-convincing comment of comparable length and yields a
sample of 1,235 comments. For each of these two samples, and for the full, unmatched dataset, the
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
illustration of a general methodological risk — model re-selection on small samples — rather than
as evidence against re-selection as a practice in general.

A second, separately constructed length-matched sample is introduced later in this report, within
the rhetorical-scoring subset rather than the full dataset (§2.7). It is important to state clearly,
here and wherever either sample's results are reported, that these are two distinct constructions
answering two distinct questions, and that figures computed on one must not be combined with
figures computed on the other as though they came from a single experiment. The sample described in
this section is built from the full 3,971-comment dataset using strict one-to-one length pairing,
and it measures each individual model's own lift over no-skill, before and after matching. The
sample described in §2.7 is built, separately, from the 2,753-comment rhetorical-scoring subset
using decile-bin matching on word count, and it measures something narrower: not any individual
model's overall lift, but specifically the *incremental* gain contributed by adding the eight
strategy scores to an existing feature set, before and after that gain is recomputed on length-
matched data.

[TABLE PLACEHOLDER: The two length-matched sample constructions side by side — source notebook,
base population, matching procedure, resulting sample size, and the specific question each is used
to answer.]

## 2.6 Rhetorical-Strategy Scoring

The length-controlled analysis in §2.5 establishes that some non-length signal survives across
several modeling approaches, but it does not characterize what that signal consists of. To
investigate whether the manner in which an argument is constructed, rather than simply how much is
written, carries predictive information, eight rhetorical strategies relevant to CMV-style
argumentation were defined: justification, concession, evidence, refutation, personal narrative,
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
Any percentage computed within this scored subset in this report or elsewhere should be understood
against this different, smaller-population denominator, and should not be compared directly to a
whole-dataset percentage without noting the distinction.

The scoring prompt went through one documented revision. An initial version presented each comment
to the LLM without including the original post it was replying to. Review of this initial pass
identified that two of the eight strategies — concession and refutation — are defined relative to
a specific point made in the original post, and cannot be judged accurately without the LLM having
access to that point. A revised prompt batches comments by original post and includes the post's
own text as context. In a manually reviewed comparison sample, refutation scores increased for the
majority of comments under the revised prompt (6 of 14 reviewed rows increased, 1 decreased),
consistent with the earlier, original-post-blind prompt having systematically under-detected
refutation. The full scoring run used the revised, original-post-aware prompt.

Two reliability checks were carried out, both explicitly described by the author as lightweight
checks rather than a formal inter-annotator-agreement study. The first is a test-retest check: 200
comments were scored twice, independently, and the two passes were compared. Agreement was strong
across all eight strategies, with 95–99% of repeat scores falling within one point of the original
score (mean absolute difference between 0.14 and 0.41 depending on strategy). The second is a
comparison against a single human judge's hand-scoring of 18 comments (9 convincing, 9 not).
Agreement with this human judge was considerably weaker than the test-retest figures, and varied
substantially by strategy; it was weakest specifically for refutation and justification, whose
initial exact-match agreement with the LLM was 16.7% and 11.1% respectively. After the judge
reviewed her own first-pass scores in light of the disagreements and revised some of them, exact-
match agreement improved across all eight strategies (refutation to 27.8%, justification to
22.2%), but refutation and justification remained the two hardest strategies to calibrate against
human judgment even after this review, with the highest mean absolute difference (approximately
0.89) of the eight strategies. This reliability pattern is directly relevant to results reported
later in this project, since refutation and justification also prove to be the two strategies most
central to the project's predictive and interpretive findings (§2.8); this connection is stated here
because it should be considered alongside those findings, not treated only as a separate,
after-the-fact limitation.

[TABLE PLACEHOLDER: The eight rhetorical strategies, with a one-sentence definition for each, and
the test-retest and human-agreement reliability figures for each strategy.]

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
§2.3–§2.4, but a related, narrower representation built specifically for this analysis; see
`METHODS_REVIEW_NOTES.md` for a fuller account of this distinction and an open question about it.
Combined with the eight strategy scores, the resulting representation totals 237 columns.

The engineered-feature baseline alone achieves PR-AUC 0.4467 (± 0.0196, see below); adding the
eight strategy scores raises this to 0.5202 (± 0.0177), a gain of 0.0735. The eight strategy scores
alone, with no engineered features at all, achieve PR-AUC 0.3538 (± 0.0079) — a real, standalone
predictive signal, positioned between this population's own no-skill baseline (0.2382) and the
combined 237-column model.

A methodological finding from this stage of the analysis is reported here because it applies to
every PR-AUC figure from this point in the project onward. XGBoost constructs each decision tree by
selecting, at every split, whichever candidate feature yields the greatest reduction in training
loss; when two or more features are correlated with one another — as many of the 237 columns in
this representation are — their split-gain values can be close to tied, and such ties are broken
deterministically according to each feature's position (column order) in the input matrix.
Refitting the identical 237-column representation eight times, changing only the order in which
columns were presented (the original ordering together with seven random reshufflings — the same
number of orderings used for every mean/standard-deviation figure reported throughout this and the
following section, for consistency), produced PR-AUC values ranging from 0.4908 to 0.5390 — a range
of 0.0482, comparable in size to some of the effects under investigation in this project. This
sensitivity was confirmed not to be an artifact of SMOTE, whose oversampling procedure operates on
Euclidean distances between feature vectors and does not depend on the order in which columns are
arranged. In light of this finding, every PR-AUC value reported from this point in the project
onward, including the figures given above, is computed as a mean and standard deviation across
eight random column orderings of the same underlying feature matrix, rather than as a single model
fit. These standard deviations describe sensitivity to an essentially arbitrary implementation
detail — column order — and are not bootstrap confidence intervals or any other estimate of
sampling uncertainty; they should not be interpreted as such.

A further, parsimony-oriented finding is reported here because it bears on how the engineered and
strategy-score features interact. Among all feature combinations evaluated, the smallest — the
eight strategy scores together with `word_count` alone (nine columns in total) — achieved the
highest PR-AUC of any configuration tested in this project, 0.5379 (± 0.0109), exceeding the full
237-column representation. Re-running the classifier and resampling grid restricted to these nine
columns found that a random forest classifier performed comparably to the originally selected
XGBoost configuration (0.5432 ± 0.0065 versus 0.5380 ± 0.0108) — a difference smaller than either
classifier's own order-induced spread, and therefore not a basis for preferring one architecture
over the other on this particular feature set.

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
to be resolved (or left open) in Results and Discussion, rather than papered over by relying on only
one of the two measures.

[TABLE PLACEHOLDER: SHAP mean |value| and Mann–Whitney rank-biserial r for each of the eight
strategies, full population and length-matched sample side by side, with FDR-significance flags.]

[FIGURE PLACEHOLDER: SHAP summary (beeswarm) plots for the nine-feature model, full population and
length-matched sample, or a simplified ranked comparison if the full beeswarm plots are not
sufficiently legible for print.]

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
WA results should be read rather than being a matter solely for the Discussion chapter: this
analysis supports the broader claim that rhetorical-strategy scores carry predictive information in
a second, independently constructed dataset, but it does not, by itself, establish that their
*incremental* predictive contribution specifically beyond comment length replicates to the same
extent documented for the home dataset in §2.7. No length-controlled ablation comparable to the one
described there was performed on WA; establishing that stronger, more specific claim would require
such an analysis, which falls outside the scope of the present external-validation check.

[TABLE PLACEHOLDER: Home dataset versus Winning Arguments, side by side — construction method,
size, negative-example construction, and no-skill baseline for each.]
