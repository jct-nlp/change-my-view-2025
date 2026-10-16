# Readiness Checklist — M.Sc. Report & Presentation

Status as of 2026-10-16 (figures rows updated; all other rows unchanged from 2026-10-14 unless
noted). Classification: **Ready** / **Done** / **Needs writing or editing** / **Needs
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
| Methods — external validation | **Ready** | No | Full numeric chain extracted from `08`, including the author's own explicitly-labeled untested hypothesis. **Corrected 2026-10-14:** the WA procedure is not zero-shot transfer — the architecture/configuration selected on the home dataset is reused unchanged, but a new model is fit on WA's own training split and evaluated on WA's own heldout split. Re-verified directly against `08_wa_external_validation.ipynb` cell 19. See `REPORT_OUTLINE.md` §3.9 for the precise wording to reuse. |
| Results — all findings | **Ready** | No | Eight findings fully drafted with exact numbers and sources in `RESULTS_SHORTLIST.md`; writing the Results section is mostly assembling and sequencing these, not researching them further. |
| Results — figures/visualizations | **Done** | No | All seven essential figures built, reviewed through two refinement passes, and approved (2026-10-16). See dedicated row below. |
| Discussion — conclusions/insights | **Needs writing or editing** | No | Content exists (`REPORT_OUTLINE.md` §5); needs to be written as connected prose, not a bullet list. |
| Discussion — deep-learning techniques | **Ready** | No | Full list of transformer/LLM-based methods used (and in two cases rejected) already identified. |
| Discussion — limitations | **Ready** | No | Comprehensive list already assembled, including the refutation/justification reliability caveat. |
| Discussion — suggested improvements | **Ready** | No | Three concrete, explicitly-labeled-as-future-work items already identified; must be written as proposals, not completed work. |
| References | **Needs writing or editing** | No | The four citations are verified (see dedicated rows below) but not yet formatted into an actual References-chapter bibliography. Citation style resolved: author-year (2026-10-17). See the task item below. |
| Code/data access link | **Ready**, pending one decision | No | README.md already has complete, usable data-access instructions; needs a decision on whether to pin a specific commit/release for reproducibility (repo is still actively changing). |

## Cross-cutting items

| Item | Status | Needs new experiments? | Notes |
|---|---|---|---|
| **All seven essential presentation figures** | **Done** | No | All seven figures from `FIGURE_SELECTION_PLAN.md`'s shortlist are built, documented, and approved. `degree_completion/figures/` holds PNG (300 dpi) + SVG for each, generated reproducibly by `make_figures.py` from live notebook cell reads (not copied from prior documentation), with full source trail, exact values, population/split, metric definitions, and speaker notes in `figures/FIGURE_NOTES.md`. Went through two review-and-refine passes on 2026-10-15 and was reviewed and approved by you on 2026-10-16. No notebook, dataset, or model output was modified to produce them. |
| Two embedded figures in `02_data_annotation.ipynb` (cells 16, 31) | **Needs verification — genuinely outstanding** | No | Content unknown — image data was not extracted during the research pass behind these documents (by design, to control context usage). Open the notebook directly to see what they show before deciding whether to use them. **Not** among the seven approved figures above — these are separate, still-unopened notebook cells. |
| Correlation heatmap at end of `03.3_feature_engineering_op_relation.ipynb` | **Needs verification — genuinely outstanding** | No | Same reason as above; also not among the seven approved figures. |
| Citation: Labruna, Modzelewski, Satta & Da San Martino (2026), Findings of EACL 2026 | **Verified** | No | *(Updated 2026-10-17 — this row was stale.)* Confirmed directly against the published paper on 2026-10-14 (ACL Anthology 2026.findings-eacl.97): title, author order, venue, and page numbers (1888–1915) all match. Table 1 benchmark numbers quoted in `08_wa_external_validation.ipynb` cross-checked value-by-value and match exactly. See `PROGRESS.md`'s 2026-10-14 entry. Still outstanding: formatting this into an actual bibliography entry — see the new task item below. |
| Citation: Tan, Niculae, Danescu-Niculescu-Mizil & Lee (2016), WWW | **Verified** | No | *(Updated 2026-10-17 — this row was stale.)* Confirmed directly against the published paper on 2026-10-14 (WWW 2016, pp. 613–624); the 4,263-pair dataset construction independently matches `08`'s own `assert len(pairs_df) == 4263`. See `PROGRESS.md`'s 2026-10-14 entry. Still outstanding: formatting this into an actual bibliography entry — see the new task item below. |
| Refutation reannotation (second human-agreement check on `refutation` scoring) | **Out of scope — not a task right now** | N/A | Per explicit user instruction (2026-10-14): ignore `refutation_reannotation/` entirely at this stage; it is not part of the current degree-completion work. (For the record, both judging sheets are still 100% empty on disk — but this is noted only so it isn't mistakenly cited as a finding, not as an open task to schedule.) **Do not cite it as a completed, in-progress, or planned validation step anywhere in the report.** |
| AI tools used (References requirement) | **Needs writing or editing** | No | Confirmed: this is the university's own requirement, not a journal-specific one — the instructions given for this report explicitly list "AI tools used" under item 6 (References). In scope and already covered: Gemini as the project's research-method tool (feature/strategy scoring, extensively documented in `REPORT_OUTLINE.md`/`RESULTS_SHORTLIST.md`). Remaining open question is narrower than previously framed — not *whether* to disclose, but whether the same line item should also name any AI coding assistant used while building the pipeline/notebooks themselves, alongside Gemini's research-method role. |
| README's stated course context ("final project in Text Mining curse at the Jerusalem College of Technology, 2025") | **Needs writing or editing** | No | This line in `README.md` reflects the project's original framing, now superseded by the M.Sc. degree-completion scope. Not a blocker for anything in `degree_completion/`, but worth updating in `README.md` itself at some point so the repo's own front door doesn't contradict the report. |
| `wa_quick_experiments.ipynb` findings in the final report | **Resolved — out of scope** | No | Per explicit instruction (2026-10-14): excluded from the official report and presentation. The notebook self-describes as scratch/exploratory, and one of its three checks ("Experiment 2") is genuine zero-shot transfer (home-trained model applied to WA with no refitting) — not to be confused with `08`'s own WA procedure, which refits a new model on WA's training split. Retained only as a labeled, out-of-scope entry in `RESULTS_SHORTLIST.md` for the research record, not as a reportable finding. |

## What can be done entirely without new experiments, starting now

Everything in both tables above is unblocked — `refutation_reannotation/` is explicitly out of
scope for now rather than a pending task, so there is no outstanding item requiring new primary
data collection, a new experiment, or a model run. Concretely, the fully unblocked work right now
is:

1. Writing the Abstract, Introduction, Methods, Results, and Discussion prose from the material
   already assembled in `REPORT_OUTLINE.md` and `RESULTS_SHORTLIST.md`.
2. ~~Re-exporting/screenshotting the ~6–8 needed figures from the already-run notebooks~~ —
   **done.** All seven essential figures are built, documented, and approved; see the
   figures rows above.
3. ~~Verifying the two secondhand-cited references against their original papers.~~ — **done**
   (2026-10-14; see the citation rows above).
4. Building the pipeline-architecture diagram for the Introduction (does not exist anywhere yet,
   needs to be drawn fresh from the already-documented 9-stage pipeline).
5. Drafting presentation slides from `PRESENTATION_STORYLINE.md`.
6. Writing the References section's "AI tools used" line for Gemini's research-method role (already
   fully documented); confirming only the narrower question below before finalizing it.
7. **Formatting the four already-verified citations, the code-reuse credit, and the
   systems/libraries list (all listed in `REPORT_OUTLINE.md` §6) into an actual References-chapter
   bibliography, author-year style (resolved 2026-10-17).** Not yet started — no References
   chapter exists in `report_draft/` yet (only the Introduction does). Currently only folded into
   `PROGRESS.md`'s Sat 2026-11-07 "Finish report" catch-all task, not tracked as its own item
   until now.

## Decisions requiring the user's input (collected from across all four documents)

1. **AI tools used — scope of the References line:** should it also name any AI coding assistant
   used while building the pipeline/notebooks, alongside Gemini's already-documented
   research-method role, or is Gemini-as-method the intended scope?
2. **Repository pinning** — tag/cite a specific commit for the "link to reproducible code"
   requirement, given the repo is still being actively edited?
3. ~~**Figure style** — reuse raw notebook screenshots as-is, or redraw cleanly for the
   report/slides?~~ — **resolved.** All seven figures were redrawn cleanly (not raw
   screenshots) and have been reviewed and approved (2026-10-16).
4. ~~**Two citations** (Labruna et al. 2026; Tan et al. 2016) were read secondhand from notebook
   markdown rather than the original papers — fine to proceed on that basis, or verify first?~~ —
   **resolved.** Both verified directly against the published papers (2026-10-14); see the
   citation rows above.
5. ~~**Citation style** — numbered or author-year for the References chapter?~~ — **resolved
   (2026-10-17): author-year.** Not specified by the university requirements, so chosen for
   consistency with the in-text "(Author, Year)" citations already used throughout
   `01_INTRODUCTION.md` — no rewriting of existing in-text citations needed when the References
   chapter is written.
