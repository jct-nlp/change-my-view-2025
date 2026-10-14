# Presentation Storyline — ~15 minutes

Audience assumption: understands machine learning (metrics, classifiers, confounds, train/test
splits) but has likely never heard of ChangeMyView or seen a "delta." Every CMV-specific term gets
one sentence of plain explanation before it's used.

This is a storyline, not slides — titles and timing are suggestions to build from, not a deck.

---

## Structure overview

| # | Slide title | Time | Content source |
|---|---|---|---|
| 1 | Title | 0:00–0:30 | — |
| 2 | What is ChangeMyView? | 0:30–2:00 | `00_introduction.ipynb` §1.1 (adapt from README intro) |
| 3 | The research question | 2:00–3:15 | `00_introduction.ipynb` §1.2 |
| 4 | Dataset & pipeline | 3:15–5:15 | README notebook table; `02_data_annotation.ipynb` |
| 5 | Finding 1 — A length confound | 5:15–7:15 | `04.2_baseline_comparison.ipynb` |
| 6 | Finding 2 — Does rhetoric survive length control? | 7:15–9:45 | `06`, `07_strategy_interpretation.ipynb` |
| 7 | Finding 3 — External validation: generalizes, with a twist | 9:45–11:45 | `08_wa_external_validation.ipynb` |
| 8 | Conclusions & limitations | 11:45–13:30 | Discussion material, `REPORT_OUTLINE.md` §5 |
| 9 | Backup slides (held in reserve) | — | Findings 6–8 from `RESULTS_SHORTLIST.md` |

Total: ~13.5 minutes of core content + 1.5 minute buffer for pacing/questions mid-talk, which is
realistic for a first run-through — the backup slides are the first thing to cut if running long,
not something to compress into the main flow.

---

## Slide 1 — Title (0:00–0:30)

Project title, your name, degree program. One line stating the two-part scope: "predicting
persuasive comments on Reddit, and testing whether a defined set of rhetorical strategies explains
that persuasion beyond simpler signals."

---

## Slide 2 — What is ChangeMyView, and what's a "delta"? (0:30–2:00)

**Explain, in plain terms, assuming zero prior familiarity:**
- r/changemyview is a public debate forum: someone posts an opinion they're open to changing,
  others reply trying to change it.
- The "delta" (∆) is the forum's own built-in success label: if the original poster is genuinely
  persuaded by a reply, they award that reply a delta. This is not something the project invented
  or inferred — it's a mechanism the community itself uses and has used for years.
- This is what makes CMV useful for this kind of research: it gives a free, built-in label for
  "this specific reply was persuasive," at the level of an individual comment, not just a vague
  sense of thread outcome.
- One sentence bridging to the project: "we treat predicting which comments get a delta as the
  core machine-learning problem."

**Recommended figure:** none needed — this should be the one slide that's pure explanation, kept
visual with maybe a simple annotated screenshot-style mockup of a post + a delta-awarded reply (not
an existing figure from any notebook — would need to be created fresh, or described verbally).

---

## Slide 3 — The research question (2:00–3:15)

State the project's actual scoped goals (from `00_introduction.ipynb` §1.2), compressed to 3
bullets:
1. What textual properties distinguish persuasive from non-persuasive replies — including the
   role of simple comment length?
2. Does a small, explicitly defined set of rhetorical strategies (e.g. directly refuting a flaw in
   the original argument, or giving a reason/justification) add predictive power beyond simpler
   features?
3. Do the answers hold up on an independent dataset built a different way?

**One sentence of motivation:** practical relevance — automated debate analysis, content
moderation, persuasive-writing assistance — without over-dwelling on it; the audience's attention
should go to the method and findings, not the motivation.

---

## Slide 4 — Dataset & methodology (3:15–5:15)

**Cover, concisely:**
- Source: 500 top r/changemyview posts via the Reddit API, parsed into one row per *reply*
  (comment), each one either delta-awarded (persuasive) or not. Final dataset: **3,971 comments,
  641 persuasive (16.1%)** — verified directly from `results/02_comments_df_base.csv.zip`.
- Feature engineering: multiple families tried per signal (sentiment, tone, style, rhetorical
  appeals), several lexicon/classifier methods tried and **rejected** after failing qualitative
  spot-checks, replaced with LLM-based scoring (Gemini) — worth one sentence on this, it's a real
  methodological decision, not an afterthought.
- Evaluation discipline: thread-grouped train/test split (no comment from the same discussion
  thread appears in both train and test — otherwise the model could "peek" at related comments),
  and PR-AUC as the primary metric (because only ~16% of comments are persuasive, so plain accuracy
  would be misleading).

**Recommended figure:** a simple pipeline diagram — data collection → annotation → feature
engineering → modeling → strategy scoring → external validation. **Does not exist yet as a figure
in any notebook** — build one fresh for this slide (see `REPORT_OUTLINE.md` §2.5).

---

## Slide 5 — Finding 1: a length confound (5:15–7:15)

**The story, in order:**
1. Build a careful feature-engineered model. It gets PR-AUC 0.3848 on held-out test data.
2. Sanity-check it against a bare-minimum baseline — just TF-IDF word counts, no engineered
   features at all. The baseline wins: PR-AUC 0.5119.
3. Dig into why. The TF-IDF model's top predictive words are mostly stopwords and the word "edit"
   — a sign it's really tracking comment *length*, not content.
4. Test that directly: a model using *only* word count (one number per comment) scores PR-AUC
   0.5256 — beating everything else so far. Persuasive comments average 217 words; non-persuasive
   ones average 78.
5. The fix: add word count into the engineered model. It closes most of the gap (PR-AUC 0.3848 →
   0.4922).

**Framing line:** "this wasn't a failure — catching this is what let the rest of the project ask a
sharper question: is there anything *beyond* length that matters?"

**Recommended figure:** the word-count-by-outcome histogram from `04.2_baseline_comparison.ipynb`
— exists as an embedded notebook output, needs to be re-exported/screenshotted before use (no
standalone image file exists anywhere in the repo currently).

---

## Slide 6: Finding 2 — does rhetoric survive once length is controlled? (7:15–9:45)

**The story:**
1. Score every comment on 8 specific, explicitly-defined rhetorical strategies (e.g.
   **refutation** — directly countering a specific flaw in the original argument; **justification**
   — giving a reason beyond bare assertion), 1–5 scale, using an LLM, following a method adapted
   from recent persuasion-detection literature (Labruna et al., 2026).
2. Adding these 8 scores to the model produces a real gain: PR-AUC 0.4467 → 0.5202. The best
   feature set found in the *entire* project is just these 8 scores plus word count — 9 features,
   beating a 237-feature version.
3. The critical test: does this gain survive once length is matched between classes (so it can't
   just be "longer comments score higher on strategies")? Yes — the strategy gain keeps 84% of its
   value under strict length-matching, versus only ~18–25% for word count or the engineered
   features alone.
4. Drilling into *which* strategy: **refutation** holds up consistently (top or near-top by two
   independent interpretability methods, even after length-matching). **Justification** has the
   single largest raw effect — but most of that evaporates once length is controlled, meaning it
   was largely tracking length, not genuine rhetorical content.
5. **One caveat, stated directly, not hidden:** refutation and justification are also the two
   strategies the LLM scorer agrees with a human judge on least reliably. The strongest finding and
   the shakiest measurement sit on the same two strategies — worth saying so.

**Recommended figures:** the two SHAP beeswarm plots (full population vs. length-matched) from
`07_strategy_interpretation.ipynb`, shown side by side — the clearest single visual argument in
the whole project for "refutation holds up, justification doesn't." Both exist only as embedded
notebook images; re-export before the talk.

---

## Slide 7 — Finding 3: external validation (9:45–11:45)

**The story:**
1. Re-run the same rubric, same prompt, same model architecture — completely unchanged, no
   re-tuning — on an independent dataset: the Winning Arguments corpus (Tan et al., 2016), built
   with a different sampling method than this project's own data.
2. The structural finding replicates: rhetorical strategies again add real predictive value beyond
   length, and a small 4-feature "robust core" model lands between two published baseline tiers
   from the literature (above a simple independent-scoring baseline, below the full published
   method) — a reasonable, honestly bounded result.
3. But the *detail* doesn't replicate cleanly: on this new dataset, **justification** is the
   dominant strategy, not refutation — the opposite ranking from the home dataset.
4. One open, explicitly-labeled hypothesis for why (not a tested conclusion): the two datasets
   build their "non-persuasive" comparison examples differently — this project's own negative
   examples aren't filtered for topical relevance the way the external dataset's are, which may be
   inflating `justification`'s apparent length-independence specifically on the external data.

**Framing line:** "a partial replication is still a meaningful result — it tells us precisely
which part of the finding is robust (rhetoric matters beyond length) and which part was specific to
this dataset (which strategy, exactly)."

**Recommended figure:** none strictly required if time is tight; if including one, the WA
feature-set comparison table (PR-AUC/pairwise-accuracy) is more presentation-friendly than another
SHAP plot at this point in the talk — two SHAP slides back-to-back risks losing the audience.

---

## Slide 8 — Conclusions & limitations (11:45–13:30)

**Conclusions (state plainly, in this order of confidence):**
1. Comment length is the single dominant predictive signal in this dataset — true, consistently,
   everywhere it was tested.
2. A real, length-independent rhetorical signal exists beyond that — also true, and replicates
   structurally on an independent corpus.
3. That signal concentrates specifically in `refutation` on this project's own data — true here,
   but did not replicate in that specific form externally; `justification` took its place.

**Limitations (be direct, don't bury these):**
- Single subreddit, single platform, English only.
- The rhetorical-strategy scores are LLM-generated; their reliability is weakest on exactly the two
  strategies (`refutation`, `justification`) that matter most for the project's headline claims —
  be upfront about this if asked.
- Most individual point-estimate comparisons in this project have overlapping confidence intervals
  at this sample size; the strongest, most decisive comparisons are the ones called out explicitly
  in this talk (the word-count discovery, the strategy-ablation gain), not every number shown.

**Suggested improvements (explicitly framed as future work, not completed):** directly test the
negative-sampling hypothesis from Finding 3; validate on a third, differently-built corpus.

---

## Backup slides (held in reserve, not part of the core 13.5 minutes)

- **A 9-feature model nearly matches the full 237-feature pipeline** (Finding 8 in
  `RESULTS_SHORTLIST.md`) — good if asked "could this be simplified?"
- **Zero-shot cross-corpus transfer** (Finding 6) — good if asked "does the model generalize
  without retraining?" Answer: above chance, but clearly worse than refitting.
- **The column-order sensitivity finding** (buried inside Finding 3) — good if asked "how
  stable/reproducible are these PR-AUC numbers?" Answer: report as mean±std across orderings, here's
  why.

---

## Possible examiner questions, with where the answer lives

| Question | Where the answer is |
|---|---|
| "How do you know the model isn't just learning length again when you say strategies matter?" | Finding 2/Slide 6 — length-matched SHAP comparison, 84% retention number. |
| "Is refutation really the key strategy, or is that just this one dataset?" | Slide 7 — explicitly: no, it reverses on WA; justification dominates there. |
| "How reliable is the LLM's rhetorical scoring?" | `RESULTS_SHORTLIST.md` Finding 7 — self-consistency strong, human agreement weaker, worst exactly on refutation/justification. |
| "Why PR-AUC and not accuracy or ROC-AUC?" | Slide 4 — ~16% positive rate; justified explicitly in `04.1_model_training.ipynb`. |
| "Are these results statistically significant?" | Be honest: most point comparisons have overlapping bootstrap CIs; the word-count and strategy-ablation gains are the two that clear that bar most clearly. |
| "Why didn't you just use a bigger/transformer model end-to-end instead of hand-built features?" | Backup slide — a 9-feature interpretable model nearly matches a 237-feature one; embeddings added on top barely moved pairwise accuracy on WA despite raising PR-AUC — supports an explicit, defensible interpretability-over-raw-performance choice. |
| "What would you do differently / next?" | Slide 8's suggested improvements — testing the negative-sampling hypothesis, a third corpus. |
| "Does this replicate prior published work on CMV?" | Slide 3/Related work — contrast with Priniski & Horne (2018, thread-level regression) and Na & DeDeo (2022, unsupervised pattern discovery, who found a refutation-like category to be *least* effective — a genuine point of tension worth acknowledging rather than avoiding). |

---

## Decisions needed before building actual slides

- Whether to include the "what is a delta" explanation as text or build a simple annotated mockup
  image (no such image exists yet in any notebook or file).
- Whether to show any raw notebook screenshots as-is (SHAP plots, histograms) or redraw them
  cleanly for slides — redrawing takes more time but reads better on a projector; this is a
  time-tradeoff call, not a correctness one.
