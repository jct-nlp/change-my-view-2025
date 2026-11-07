# 4. Discussion

## 4.1 RQ1 and RQ2: What predicts persuasive success, and which strategies carry signal beyond length

Comment length is the single strongest predictor of delta-award status on the home dataset. Both a
plain bag-of-words baseline and `word_count` alone outperform the full
engineered-feature pipeline (§3.1.2), and much of the apparent predictive advantage of the
engineered features is attributable to their association with length, not to rhetorical
content specifically (§3.1.5). This is an associative, not causal, finding: it establishes that
length is informative and that the engineered features did not capture that information as well as
length itself does, not that length causes persuasive success.

A real signal survives rigorous length control nonetheless (§3.1.4), and this is the empirical basis
for RQ2. Adding the eight rhetorical-strategy scores to the engineered baseline produces a real,
order-robust gain (0.4434 to 0.5098 PR-AUC, +0.0664, §3.2.2) that retains 84% of its value under
strict length matching (0.0664 to 0.0557, §3.2.3) — markedly more robust to length control than
either `word_count`'s own lift or the full engineered pipeline's lift, both of which retain only
about 18% of their full-dataset values under the analogous match (§3.1.4). This is the project's
central empirical conclusion for RQ2: an explicitly defined rhetorical-strategy rubric captures
something beyond comment length that conventional engineered linguistic features, on their own, did
not.

### Resolving the SHAP/rank-biserial disagreement

Within that block-level finding, two interpretive measures disagree about which single strategy is
"most important," even before length is controlled for: SHAP ranks `refutation` first (0.6034),
while the rank-biserial effect size ranks `justification` first (0.3584, ahead of `refutation`'s
0.3149) (§3.2.5). This disagreement is not a contradiction to resolve by choosing a winner — it
reflects two measures answering genuinely different questions, and the distinction is treated here
as a conclusion in its own right, not left open.

The rank-biserial effect size is a model-free measure of marginal association: it asks how strongly
a single strategy's score differs between convincing and non-convincing comments, considered in
isolation from every other feature. SHAP importance is a model-dependent measure of predictive
attribution: it asks how much the *fitted classifier* actually relies on a given feature once every
other input — including features correlated with it — is already present in the model. A feature
can score highly on one measure without scoring highly on the other precisely because the two
measures are not estimating the same quantity, and disagreement between them does not indicate that
either computation is wrong.

Concretely, `refutation`'s SHAP advantage means that, for this specific fitted XGBoost pipeline, once
`word_count` and the other seven strategies are already in the feature matrix, the model relies more
heavily on `refutation` than on any other strategy to resolve the cases its other features leave
ambiguous. `justification`'s rank-biserial advantage means that, considered as an isolated pairwise
relationship, `justification`'s scores covary with delta-award status more strongly than any other
strategy's scores do on their own. Both statements are compatible: a feature can carry the strongest
raw association with the outcome while contributing comparatively little *additional* information to
a model that already has other, partially redundant, predictors available, and vice versa. On this
basis, this report does not collapse the full-population analysis into a single "most important
strategy." `Refutation` is the strongest model-attributed feature in this specific fitted pipeline;
`justification` is the strategy whose score differs most, on its own, between convincing and
non-convincing comments. Neither claim should be read as a general or causal statement about which
rhetorical strategy is most persuasive on CMV, and the two should not be merged into one ranking.

The two strategies diverge sharply once length is controlled for, and this divergence is informative
independent of which baseline measure one prefers. `Refutation` retains the large majority of both
measures under length matching — SHAP declines only modestly (0.6034 to 0.5193, holding the top rank
among strategies) and the rank-biserial effect size retains 49% of its value (0.3149 to 0.1540,
remaining significant). `Justification`'s SHAP importance, by contrast, collapses from the model's
third most influential input to one of its least influential strategies (0.4156 to 0.1956) — most of
what the fitted model was using `justification` for on the full population appears length-related —
while its rank-biserial effect size retains only 37% of its value (0.3584 to 0.1309) but remains
statistically significant after FDR correction (§3.2.6). The conclusion drawn here is therefore
specific, not global: `refutation` is the home dataset's most length-robust rhetorical strategy by
*both* interpretive measures, even though the two measures disagree about which strategy is most
important before length is taken into account.

### The `personal_narrative` sign flip

One further pattern deserves comment, not just a bare mention. `Personal_narrative`'s
rank-biserial effect is not statistically significant on the full population, but is significant and
*negative* on the length-matched sample (r = −0.0859, §3.2.6) — the only strategy whose significance
status and sign both shift between the two populations. One plausible interpretation is that length
is masking a genuine negative association in the full population: a tendency for heavy
personal-narrative use to co-occur with lower persuasive success may be offset, on average, by a
correlated and opposing length effect, becoming visible only once length variation is reduced. This
is offered here explicitly as one candidate interpretation, not an established explanation. The
length-matched sample is a separately and specifically constructed subset of the data (§2.5, §3.2.3),
not simply a length-corrected version of the same population with every other property held fixed, so
the comparison does not isolate length as the only variable that changed between the two analyses. An
equally available alternative is that the full-population effect is simply not robust and the
length-matched result is closer to noise. Distinguishing between these honestly would require further
analysis beyond this project's scope; the result is reported here as a pattern worth noting, not as a
resolved finding.

## 4.2 RQ3: Cross-corpus evaluation — what replicates, what does not, and why that matters

The external evaluation on Winning Arguments (WA) supports two structural conclusions cleanly, on an
independently constructed corpus built with a different negative-sampling methodology: rhetorical-
strategy scores carry predictive signal beyond `word_count`, and a small, pre-specified strategy-based
feature set is competitive with, or exceeds, larger combinations (§3.3.2, §3.3.5, §3.3.6).

This replication deserves substantial weight as independent-corpus support for the central RQ2 claim.
WA's own length-matched robustness check (§3.3.2) found the strategy-inclusive combination retaining
87% of its lift over no-skill (0.0523 to 0.0455) while `word_count` alone retained only 44% (0.0476
to 0.0211) — the same qualitative pattern already established on the home dataset (84% versus roughly
18%, §3.2.3, §3.1.4). That a second corpus, built by a different team with a deliberately different
sampling procedure, reproduces the same qualitative asymmetry is meaningful evidence that this
project's central claim — rhetorical-strategy scores are not simply a proxy for comment length — is
not an artifact specific to how this project's own dataset happens to be constructed.

That said, the strength of this conclusion should be calibrated to what the check actually measures.
WA's own length confound is considerably weaker to begin with: matching reduces `word_count`'s own
correlation with the outcome by only 23% on WA, against a 79% reduction under the home dataset's
analogous match (§3.3.2) — WA's winning and losing replies are already fairly similar in length
before any matching is applied, since the corpus's own sampling is topic-matched instead of
unfiltered. The WA check is therefore a somewhat less demanding test than the one the home dataset's
own ablation poses, and it compares a narrower pair of configurations (`word_count` alone against
`word_count` plus strategy scores), not the home dataset's full engineered-pipeline-against-
itself-plus-strategies comparison, since WA was never scored on that pipeline (§2.9, §3.3.2). The
appropriate conclusion is that the strategy signal recurs qualitatively on a second corpus under a
narrower and easier test, not that its magnitude or its independence from length has been
established to the same degree as on the home dataset.

The specific claim that `refutation` leads that signal, and that `justification` is specifically the
strategy whose importance is length-confounded, does not replicate. On WA, the two strategies'
relative roles are substantially reversed: `refutation` falls from first to seventh of eight by SHAP
and from second to fourth by rank-biserial effect size, while `justification` rises to first by SHAP
and remains first by rank-biserial in both datasets — and, unlike at home, `justification`'s
rank-biserial effect does *not* collapse under WA's own length matching, retaining 84% of its value
(0.1962 to 0.1648) against the 37% retained at home (§3.3.4). This reversal does not appear to be a
measurement artifact: a dedicated test-retest check found `refutation` scored *more* consistently on
WA (76.5% exact match) than on the home dataset (63.8%, the weakest of the eight strategies there),
and a column-order-shuffle check confirms `refutation`'s low WA standing is stable across eight
repeats, never ranking above sixth of eight (§3.3.4).

A plausible, but explicitly untested, explanation is that the two corpora's negative-example
construction differs in a way that matters for which strategy looks most informative. The home
dataset's non-convincing comments are the thread's own unfiltered population; WA's non-winning
replies are deliberately selected to be topically similar to the winning reply within the same thread
(§2.9, §3.3.1). If `refutation`'s apparent importance at home partly reflects its ability to
distinguish a persuasive reply from topically *unrelated* or off-target replies — a distinction WA's
sampling procedure specifically removes by construction — this could account for its collapse on WA
without requiring any change in how persuasion itself works. This is offered as one candidate
hypothesis consistent with the observed pattern, not as a tested or established cause; no analysis in
this project isolates the sampling-methodology variable directly, and other explanations (e.g.,
differences in topic distribution, thread structure, or the populations' own base rates) have not
been ruled out either.

Taken together, this points to a specific, bounded conclusion for RQ3: the general claim that
rhetorical-strategy scores add predictive information beyond length generalizes across two
independently constructed CMV-adjacent corpora; the specific claim that one particular strategy
drives that signal does not generalize, and may instead depend on properties of how a dataset's
negative examples are selected. That dependency is itself a substantive empirical observation about
the limits of strategy-level generalization in this research area, not merely a limitation of this
project's own results.

## 4.3 Methodological insights

A recurring pattern across this project is worth naming explicitly as a learned concept, not just as
background to the final numbers. Three times, an initial result was taken seriously enough to be
tested against itself and, in each case, revised before being reported as found: the discovery that
`word_count` alone outperforms the full engineered pipeline, which prompted the length-controlled
analysis in the first place (§2.4, §3.1); a small-validation-set overfitting failure caught when model
selection was blindly re-run on the length-matched sample, where the re-selected architecture scored
*below* its own no-skill baseline on test while the originally selected architecture continued to
score positively (§2.5); and the discovery that this project's XGBoost+SMOTE pipeline is sensitive to
the order of columns in a correlated, 200+-column feature matrix, large enough (a 0.0482 PR-AUC swing
in a direct check, §2.3) to rival some of the effects under study. Each of these was disclosed and
corrected, not absorbed silently, and the third prompted an explicit methodological convention
adopted mid-project: every PR-AUC reported from that point onward is a mean ± SD across multiple
column orderings, not a single-order point estimate. This project treats that pattern — catching and
correcting problems that could have inflated or mischaracterized a result — as a positive indicator of
methodological rigor, not an admission of weakness, and the corresponding uncertainty discipline it
produced (§4.4, below) should be read as a direct consequence of it.

It is also worth stating plainly where this project's deep-learning and transformer-based techniques
actually sit in the pipeline, since the final predictive architecture is not itself a neural model:
the selected classifier throughout is XGBoost with SMOTE oversampling, a gradient-boosted tree
ensemble. Deep learning is used instead as a feature-generation and representation-learning layer
upstream of that classifier. This includes an LLM (Gemini), used both for several engineered features
(sentiment, tone, style, and the ethos/pathos/logos rhetorical-appeal scores, §2.2) and, centrally, for
the entire eight-strategy rhetorical rubric that is this project's main methodological contribution
(§2.6); transformer-based sentence embeddings (`BAAI/bge-base-en-v1.5`, reduced by PCA to roughly 187
components at ≥85% variance, §2.2); two HuggingFace transformer classifiers that were evaluated and
explicitly rejected after qualitative inspection — a RoBERTa emotion classifier, two-thirds of whose
predictions collapsed to "neutral," and a zero-shot `bart-large-mnli` classifier with visible
mislabeling on spot checks — in favor of LLM scoring (§2.2); and one HuggingFace transformer-based
persuasive-language detector (`chreh/persuasive_language_detector`) that was kept as a feature.
Reporting the two rejected models alongside the two kept ones is deliberate: the choice of which
deep-learning tool to use for a given feature was itself evaluated empirically, not assumed in
advance.

## 4.4 Limitations

- The LLM-scored rhetorical strategies carry a reliability ceiling that is not uniform across
  strategies, and it is concentrated on exactly the two strategies central to §4.1's headline
  interpretation: `refutation` and `justification` have the weakest human-judge agreement of the
  eight strategies scored (83.3% and 88.9% within one point respectively, against 94.4–100% for most
  others, §3.2.1). This does not invalidate the SHAP/rank-biserial pattern discussed above, but it
  bounds how strongly that pattern should be stated — it is possible that some part of what
  distinguishes these two strategies reflects scoring noise and not a stable underlying
  difference.
- Most individual point-estimate comparisons in this project carry overlapping uncertainty bands and
  should be read as best estimates, not as statistically decisive rankings. The clearest exceptions —
  comparisons described in this report as clearing that bar decisively — are the original
  `word_count`-versus-engineered-pipeline discovery (§3.1.2) and the strategy-ablation gain itself
  (§3.2.2). The nine-column-versus-robust-core comparison (§3.2.4) is a concrete example of a
  comparison that does *not* clear this bar: the robust core's higher point estimate is not
  distinguishable from the nine-column model once their combined order-noise band is taken into
  account.
- A second, independent source of uncertainty layers on top of the first: the fitted model's PR-AUC
  is itself sensitive to the order of the (correlated, 200+-column) feature matrix's columns, a
  property unrelated to sampling error (§2.3, §2.7, §3.2.7). Both sources bear on how confidently any
  single reported number in this report should be read.
- The WA external evaluation supports the *group-level* claim that strategy scores carry signal
  beyond length, not a claim of equal strength to the home dataset's own length-controlled ablation:
  the WA length-matched check is narrower in scope (§3.3.2, §4.2 above) and operates on a corpus
  whose own length confound is considerably weaker to begin with.
- The dataset is drawn from a single subreddit, a single platform, and is English-only; thread size
  is also structurally correlated with a thread's apparent convincing rate (one delta per thread
  roughly fixed, while negative examples scale with thread branching), which is relevant to any
  analysis that aggregates at the level of the original post instead of the individual comment
  (§2.1).
- Delta-award status is an operational proxy for persuasive success, not direct evidence of a lasting
  change in belief (§1.1). Every interpretive claim in this chapter concerns what predicts this
  observable, incentive-shaped label, not persuasion as a psychological construct in general.

## 4.5 Possible improvements and future work

The following are proposed directions for future work, not analyses already carried out in this
project:

- Test the negative-example-construction hypothesis from §4.2 directly, for instance by constructing
  a topic-matched negative-sampling variant of the home dataset itself, or by stratifying WA's own
  replies by topical similarity to the winning reply and checking whether `refutation`'s standing
  shifts systematically with that property. This would isolate whether the sampling-methodology
  difference, specifically, drives the cross-corpus reversal, rather than some other difference
  between the two corpora.
- Extend external validation to a third, differently constructed persuasion corpus. A third dataset
  would help establish whether the structural claim (RQ2's core finding) generalizes beyond two
  corpora, and whether its own strategy ranking aligns more closely with the home dataset's or with
  WA's — informative either way, given the reversal already observed between the two corpora
  evaluated here.
- Conduct a more rigorous, multi-judge inter-annotator-agreement study of the rhetorical-strategy
  rubric, particularly for `refutation` and `justification`. The reliability check performed in this
  project is explicitly a lightweight check against a single judge on a small sample (§3.2.1), and a
  larger, multi-judge study would help establish whether these two strategies' weaker LLM-human
  agreement reflects genuine ambiguity in the underlying construct or is better attributed to the
  limited power of this project's own check.

## 4.6 Conclusions

Returning to the three research questions stated in the Introduction:

**RQ1.** Comment length is the dominant single predictor of delta-award status on the home dataset.
Conventional engineered textual features, evaluated on their own, add little predictive power beyond
what length already explains, and a naive comparison that does not account for length risks
attributing predictive power to rhetorical content that is substantially a length effect instead.

**RQ2.** An explicitly defined, eight-strategy rhetorical rubric captures real predictive signal
beyond comment length that this project's conventional engineered features did not. That signal is
not uniform across strategies: `refutation` is the strategy whose contribution is most robust to
length control by both of two legitimately different interpretive measures, while `justification`
carries the largest marginal association with the outcome overall but loses most of its distinct
predictive attribution once length is accounted for. These two measures — model-based attribution and
model-free marginal association — answer different questions and are not reducible to a single
ranking.

**RQ3.** The structural claim from RQ2 — that rhetorical-strategy scores add information beyond
length — replicates on an independently constructed corpus built with a different sampling
methodology. The specific claim about which strategy drives that signal does not: `refutation` and
`justification`'s relative importance substantially reverses on Winning Arguments, in a way not
attributable to scoring-reliability differences between the two corpora. A plausible, but explicitly
untested, candidate explanation is that the two corpora's differing negative-example construction
changes what a given strategy is effectively being asked to distinguish.

Across all three questions, this project's contribution is primarily empirical and analytical: a
documented account of how strongly comment length predicts persuasive success and why that matters
for interpreting any persuasion-prediction model; a systematic, length-controlled assessment of a
purpose-built rhetorical-strategy rubric's incremental value, including which specific strategies
drive that value under which measure; and an external evaluation that identifies both what recurs and
what does not when the same approach is applied to an independently constructed dataset. The project
does not claim a new learning algorithm, and its findings about individual strategies are associative
and measure-dependent rather than causal — both limitations stated here deliberately, not left
implicit.
