# Readiness Checklist — M.Sc. Report & Presentation

Status as of 2026-10-14. Classification: **Ready** / **Needs writing or editing** / **Needs
verification** / **Missing**. The last column marks whether the item can be completed without
running any new experiments or retraining any model.

## Content by report section

| Item | Status | Needs new experiments? | Notes |
|---|---|---|---|
| Abstract | **Missing** | No | Not drafted; should be written last, after Results/Discussion text exists, so it summarizes the actual final wording rather than a prediction of it. |
| Introduction — motivation | **Ready** | No | `00_introduction.ipynb` §1.3 already states this in usable form; adapt, don't rewrite from scratch. |
| Introduction — problem definition | **Ready** | No | `00_introduction.ipynb` §1.2's goal list is already precise and complete. |
| Introduction — related work | **Needs writing or editing** | No | Facts and citable findings are gathered (`REPORT_OUTLINE.md` §2.3); two citations (Labruna et al. 2026, Tan et al. 2016) need verification against the original papers, not secondhand notebook citations — see Verification row below. Prose connecting them to this project's framing still needs to be written. |
| Introduction — what's different | **Ready** | No | Four concrete contrasts already identified and verified; just needs writing up. |
| Introduction — learning-system architecture | **Needs writing or editing** | No | Narrative content exists (README table, `00_introduction.ipynb` §1.4); the diagram itself does not exist and needs to be drawn — see Missing-figures row below. |
| Introduction — selected methods | **Ready** | No | Full list with "why chosen" rationale already extracted; needs writing up. |
| Introduction — relevance/limitations | **Ready** | No | High-level version drafted in `REPORT_OUTLINE.md` §2.7; detailed version belongs in Discussion. |
| Methods — data collection/annotation | **Ready** | No | Exact numbers independently re-verified against `results/02_comments_df_base.csv.zip` in this review (3,971 rows, 641/16.1% convincing). |
| Methods — feature engineering | **Ready** | No | Full "tried→rejected→replaced" narrative already extracted with exact numbers, across `03.1`–`03.3`. |
| Methods — model training/evaluation methodology | **Ready** | No | Thread-grouped split, PR-AUC justification, 27-combination grid all documented with exact numbers. |
| Methods — baseline investigation / length confound | **Ready** | No | Full numeric chain already extracted from `04.2`. |
| Methods — length-controlled analysis | **Ready** | No | Full numeric chain already extracted from `04.3`, including the re-selection overfitting example. |
| Methods — rhetorical scoring | **Ready**, with one caveat | No | Rubric, prompt iteration, and validation numbers extracted from `05`; the *denominator shift* (16.1% vs. 23.3%) must be stated explicitly wherever both appear together — easy to get wrong, flagged in `REPORT_OUTLINE.md` §3.6. |
| Methods — strategy ablation | **Ready** | No | Numbers extracted from `06`, including the column-order-sensitivity disclosure, which should be written up as a methodological point, not omitted. |
| Methods — strategy interpretation | **Ready** | No | SHAP and Mann-Whitney numbers, full population and length-matched, extracted from `07`. |
| Methods — external validation | **Ready** | No | Full numeric chain extracted from `08`, including the author's own explicitly-labeled untested hypothesis. |
| Results — all findings | **Ready** | No | Eight findings fully drafted with exact numbers and sources in `RESULTS_SHORTLIST.md`; writing the Results section is mostly assembling and sequencing these, not researching them further. |
| Results — figures/visualizations | **Missing** | No (re-export, not re-run) | See dedicated row below — this is the single largest concrete gap. |
| Discussion — conclusions/insights | **Needs writing or editing** | No | Content exists (`REPORT_OUTLINE.md` §5); needs to be written as connected prose, not a bullet list. |
| Discussion — deep-learning techniques | **Ready** | No | Full list of transformer/LLM-based methods used (and in two cases rejected) already identified. |
| Discussion — limitations | **Ready** | No | Comprehensive list already assembled, including the refutation/justification reliability caveat. |
| Discussion — suggested improvements | **Ready** | No | Three concrete, explicitly-labeled-as-future-work items already identified; must be written as proposals, not completed work. |
| References | **Needs verification** | No | See dedicated row below. |
| Code/data access link | **Ready**, pending one decision | No | README.md already has complete, usable data-access instructions; needs a decision on whether to pin a specific commit/release for reproducibility (repo is still actively changing). |

## Cross-cutting items

| Item | Status | Needs new experiments? | Notes |
|---|---|---|---|
| **All report/presentation figures** | **Missing** | No — re-export/screenshot existing notebook outputs, don't rerun anything | Confirmed directly: no PNG/SVG/JPG files exist anywhere in the repository outside two unrelated PDFs under `previous work/`. Every figure referenced throughout `REPORT_OUTLINE.md`/`RESULTS_SHORTLIST.md` (word-count histogram, SHAP beeswarms ×2, correlation heatmap, PR/ROC curves, two unidentified figures in `02_data_annotation.ipynb`) exists only as an embedded, non-exported notebook cell output. This is one task, not eight: open each notebook once and export the needed figures as image files. |
| Two embedded figures in `02_data_annotation.ipynb` (cells 16, 31) | **Needs verification** | No | Content unknown — image data was not extracted during the research pass behind these documents (by design, to control context usage). Open the notebook directly to see what they show before deciding whether to use them. |
| Correlation heatmap at end of `03.3_feature_engineering_op_relation.ipynb` | **Needs verification** | No | Same reason as above. |
| Citation: Labruna, Modzelewski, Satta & Da San Martino (2026), Findings of EACL 2026 | **Needs verification** | No | Read secondhand from `08_wa_external_validation.ipynb`'s own markdown citation, not confirmed against the actual paper. Confirm exact title wording and page numbers before finalizing the reference list. |
| Citation: Tan, Niculae, Danescu-Niculescu-Mizil & Lee (2016), WWW | **Needs verification** | No | Same — read secondhand from notebook markdown, not confirmed against the original paper. |
| Refutation reannotation (second human-agreement check on `refutation` scoring) | **Out of scope — not a task right now** | N/A | Per explicit user instruction (2026-10-14): ignore `refutation_reannotation/` entirely at this stage; it is not part of the current degree-completion work. (For the record, both judging sheets are still 100% empty on disk — but this is noted only so it isn't mistakenly cited as a finding, not as an open task to schedule.) **Do not cite it as a completed, in-progress, or planned validation step anywhere in the report.** |
| AI tools used (References requirement) | **Needs writing or editing** | No | Confirmed: this is the university's own requirement, not a journal-specific one — the instructions given for this report explicitly list "AI tools used" under item 6 (References). In scope and already covered: Gemini as the project's research-method tool (feature/strategy scoring, extensively documented in `REPORT_OUTLINE.md`/`RESULTS_SHORTLIST.md`). Remaining open question is narrower than previously framed — not *whether* to disclose, but whether the same line item should also name any AI coding assistant used while building the pipeline/notebooks themselves, alongside Gemini's research-method role. |
| README's stated course context ("final project in Text Mining curse at the Jerusalem College of Technology, 2025") | **Needs writing or editing** | No | This line in `README.md` reflects the project's original framing, now superseded by the M.Sc. degree-completion scope. Not a blocker for anything in `degree_completion/`, but worth updating in `README.md` itself at some point so the repo's own front door doesn't contradict the report. |
| Whether `wa_quick_experiments.ipynb` findings are used in the final report | **Needs writing or editing** (decision, not a gap) | No | The notebook explicitly self-describes as scratch/exploratory. Its three findings are usable but should be labeled "supplementary" if included — a presentation choice, not a correctness issue. |

## What can be done entirely without new experiments, starting now

Everything in both tables above is unblocked — `refutation_reannotation/` is explicitly out of
scope for now rather than a pending task, so there is no outstanding item requiring new primary
data collection, a new experiment, or a model run. Concretely, the fully unblocked work right now
is:

1. Writing the Abstract, Introduction, Methods, Results, and Discussion prose from the material
   already assembled in `REPORT_OUTLINE.md` and `RESULTS_SHORTLIST.md`.
2. Re-exporting/screenshotting the ~6–8 needed figures from the already-run notebooks (no re-run
   needed — these are saved outputs, just not saved as standalone files yet).
3. Verifying the two secondhand-cited references against their original papers.
4. Building the pipeline-architecture diagram for the Introduction (does not exist anywhere yet,
   needs to be drawn fresh from the already-documented 9-stage pipeline).
5. Drafting presentation slides from `PRESENTATION_STORYLINE.md`.
6. Writing the References section's "AI tools used" line for Gemini's research-method role (already
   fully documented); confirming only the narrower question below before finalizing it.

## Decisions requiring the user's input (collected from across all four documents)

1. **AI tools used — scope of the References line:** should it also name any AI coding assistant
   used while building the pipeline/notebooks, alongside Gemini's already-documented
   research-method role, or is Gemini-as-method the intended scope?
2. **Repository pinning** — tag/cite a specific commit for the "link to reproducible code"
   requirement, given the repo is still being actively edited?
3. **Figure style** — reuse raw notebook screenshots as-is, or redraw cleanly for the report/slides
   (time-tradeoff, not a correctness question)?
4. **Two citations** (Labruna et al. 2026; Tan et al. 2016) were read secondhand from notebook
   markdown rather than the original papers — fine to proceed on that basis, or verify first?
