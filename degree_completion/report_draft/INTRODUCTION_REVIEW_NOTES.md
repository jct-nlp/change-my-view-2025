# Review Notes — Introduction Draft

Notes for reviewing `01_INTRODUCTION.md`. This is a first draft for human review, not a
finalized text.

## 1. Claims needing factual verification or citations

- **Resolved (per your instruction):** the opening sentence originally carried a
  `[CITATION NEEDED]` marker for a generic "active topic in NLP/computational argumentation"
  claim with no supporting source in the project's bibliography. No such general-field reference
  exists in this repo (checked — no `.bib` file or equivalent on this branch), so rather than
  running a literature search for one, the sentence was rephrased to make the same point using
  only the two already-verified CMV-specific sources (Priniski & Horne, 2018; Na & DeDeo, 2022),
  cited once at the end of the merged opening paragraph rather than redundantly in two consecutive
  sentences. No new citation was added.
- The four citations used (Priniski & Horne 2018; Na & DeDeo 2022; Labruna et al. 2026; Tan et
  al. 2016) were all independently verified against the published papers in a prior review pass
  (see `REPORT_OUTLINE.md` §2.3) — their title/author/venue details are solid. What is **not**
  yet done: formatting them into an actual References-chapter bibliography entry. The in-text
  "(Author, Year)" style used here assumes that chapter will exist; the citation style (numbered
  vs. author-year) for *this* report is simply undecided — `REPORT_OUTLINE.md` does not address it
  either way. *(Correction, 2026-10-17: an earlier version of this bullet incorrectly attributed a
  "CAS Overleaf bundle has no numbered `.bst`" claim to `REPORT_OUTLINE.md`. That claim belongs to
  a separate, currently out-of-scope journal-paper citation-style question — not this MSc report.
  `REPORT_OUTLINE.md` was re-checked directly and says nothing about it.)*
- No new or additional citations were invented. No notebook-derived number in this draft goes
  beyond what `REPORT_OUTLINE.md` / `RESULTS_SHORTLIST.md` already mark `[VERIFIED]`.

## 2. Places needing your judgment or approval

- **"Evolution of the research" framing (§1.2).** The draft states plainly that the length
  confound was *not* fully anticipated in the initial design and that the length-controlled /
  ablation work was a response to an empirical finding partway through the project. This is
  accurate to the notebook record (`04.2`'s own narrative), but it is a more exposed admission
  than the original `00_introduction.ipynb` text, whose §1.2 goal list already names length and
  rhetorical strategies as objectives from the start. Confirm you're comfortable with this framing
  — it is more honest about process but also more exposed than simply restating the notebook's own
  retrospective goal list as if it were the original plan.
- **Tone of the contributions paragraph (§1.4, final paragraph).** Deliberately modest wording
  ("modest and empirical rather than methodological novelties in their own right"). If you want a
  stronger claim to novelty — e.g. the 8-strategy CMV-specific taxonomy, or the explicit
  length-controlled ablation methodology itself — that is a judgment call for you, not something
  resolved by the notebooks.
- **No research questions were changed.** The three RQs are used verbatim as given. Flagging per
  the task's own instruction that any change to them requires explicit approval — none was made.

## 3. Ambiguity about contribution or novelty

- The report does not yet state, anywhere, how this project's 8-strategy rubric relates to Na &
  DeDeo's (2022) unsupervised "Relevance & Presumption" category — the closest prior-work analogue
  to `refutation` — which they found among their *least* effective categories, the opposite of
  this project's finding. `REPORT_OUTLINE.md` §2.3 flags this explicitly as "worth one explicit
  sentence in Discussion," not the Introduction — left out of this draft deliberately, per the
  task's instruction not to overload the Introduction, but worth confirming that's still the right
  home for it.
- Methodological vs. empirical contributions are distinguished in §1.4's closing paragraph, but
  only at a high level. If the Methods/Discussion chapters later want a sharper "what exactly is
  new here vs. adapted from Priniski & Horne / Labruna et al." statement, that line will need to be
  revisited once those chapters exist — the Introduction intentionally doesn't try to settle it in
  full.

## 4. Research-question → experiment mapping

| RQ | Primary notebooks | What answers it |
|---|---|---|
| RQ1 (prediction + length) | `04.1_model_training.ipynb`, `04.2_baseline_comparison.ipynb` | Initial engineered-pipeline PR-AUC vs. TF-IDF vs. word-count-only; length/outcome correlation and within-thread check. |
| RQ2 (incremental signal beyond length) | `04.3_length_controlled_analysis.ipynb`, `06_strategy_ablation.ipynb` | Length-controlled lift comparisons (04.3); strategy-score ablation gain and its own length-matched retention (06, the 84% figure). |
| RQ3 (which strategies, how stable) | `07_strategy_interpretation.ipynb`, `08_wa_external_validation.ipynb` | SHAP + rank-biserial effect sizes by strategy, full vs. length-matched (07); same scoring/model methodology applied to Winning Arguments, with the strategy-ranking reversal (08). |

Supporting/validation notebooks referenced but not themselves answering an RQ directly:
`05_rhetorical_scoring.ipynb` (defines and validates the 8-strategy rubric used by RQ2/RQ3);
`01_data_download.ipynb` / `02_data_annotation.ipynb` (construct the dataset RQ1 is evaluated on).

## 5. Limitations held for Discussion, not placed in the Introduction

To keep the Introduction focused, the following verified limitations are intentionally **not**
elaborated here and should be carried into the Discussion chapter:

- LLM-scored rhetorical strategies have a real reliability ceiling, concentrated on exactly the
  two strategies (`refutation`, `justification`) that matter most for the headline finding
  (test-retest strong at 95–99% within-1-point agreement; human agreement, N=18 single judge,
  markedly weaker and weakest on these two).
- Most individual point-estimate PR-AUC/effect-size comparisons in this project have overlapping
  uncertainty intervals; only a few comparisons (the word-count discovery, the strategy-ablation
  gain) are strong enough to call decisive.
- XGBoost+SMOTE results on the >200-column feature matrix are sensitive to feature-column order
  (up to 0.0422 PR-AUC swing from reordering alone) — every relevant number from `06` onward is
  reported as mean±SD across orderings for this reason, not as a single run.
- The WA external validation supports a strategy-level signal on a second dataset but does not
  include a length-controlled ablation comparable to `06`'s own — so the *incremental-beyond-length*
  claim should not be read as replicated on WA to the same extent as on the home dataset.
  (The Introduction draft already states this; it is listed here as a limitation that should be
  restated, with more detail, in Discussion rather than assumed settled by the Introduction's
  mention.)
- Single subreddit, single platform, English-only dataset.
- `refutation_reannotation/` remains out of scope per standing instruction and is not referenced
  anywhere in this draft.
- The exploratory `wa_quick_experiments.ipynb` notebook (including its genuine zero-shot,
  no-refit check) is excluded from this draft, consistent with its exclusion from the official
  report and presentation.

## Scope check

- No notebooks, experimental code, datasets, or model outputs were modified or run.
- No figures were modified.
- No RQ wording was changed.
- Only the Introduction was drafted — no other report chapter was started.
