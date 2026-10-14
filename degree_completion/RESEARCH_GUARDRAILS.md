# Research Guardrails — Defense Presentation and Final Report

**Purpose:** A living checklist of scientific claims, wording, methodological caveats and scope decisions. Use when writing the report, designing slides, preparing speaker notes and answering examiners. **Not** a progress checklist or an invitation to run new experiments.

## Claim boundaries

- **Prediction is not causation.** A rhetorical-strategy score may be predictive of delta-awarded comments without proving that using the strategy causes persuasion.
- **Delta is a proxy/operational label.** It records an awarded delta in CMV; it is not a complete or error-free measurement of attitude change.
- **Avoid overclaiming independence from length.** Length matching supports the existence of additional predictive signal in the tested data; it does not eliminate every confound.
- **Distinguish performance metrics and populations.** PR-AUC depends on positive-class prevalence. Do not compare raw PR-AUC across home and WA as if their baselines were identical. Give the no-skill baseline or lift where useful.
- **Thread-aware splitting matters.** State the thread-disjoint evaluation setup and avoid treating comments from one thread as independent across train and test.

## Strategy measurement and interpretation

- The eight strategy scores are **LLM-generated measurements**, each on a 1–5 rubric, not direct observations of mental states or definitive human annotations.
- Test–retest consistency and human agreement answer different questions. The human check is small (**N=18, one judge**) and is **not** a formal inter-annotator reliability study.
- **Refutation and justification** were among the least well aligned with human scoring. Mention this near any claim about their relative importance, not only in a distant limitations slide.
- SHAP importance and univariate effect size measure different things; do not call either causal influence.
- On the original dataset, refutation remains prominent after length control; justification's model-based importance decreases substantially. Avoid “refutation is universally best.”

## Model robustness and evaluation

- Some XGBoost+SMOTE experiments are sensitive to **feature-column order**. For the relevant ablation/interpretation results, use mean±SD across orderings and avoid cherry-picked runs.
- Most pairwise point-estimate differences have overlapping uncertainty intervals; distinguish strong comparisons from suggestive ones.
- Length matching changes the sample and may reduce power. State the subset and matching setup when reporting retention percentages.
- The nine-feature result (eight strategy scores + word count) is a result **among configurations tested**, not proof of a globally optimal feature set.

## External validation: WA

- **Correct protocol:** transfer the selected architecture/training configuration from the original-data experiments; **fit it again on WA training data**, then evaluate on held-out WA data. Do **not** say the original fitted model was applied unchanged, and do **not** describe the main WA experiment as zero-shot.
- Distinguish **architecture/configuration reuse** from **parameter/weight reuse** and from **hyperparameter/model selection performed on WA**. Confirm the exact steps in `this work/08_wa_external_validation.ipynb` before finalizing technical language.
- The **group-level rhetorical signal** appears again on WA, but the **individual strategy ranking changes**: justification, not refutation, is dominant there. Describe this as partial replication.
- A possible explanation is the datasets' different **negative-example sampling procedures**. This is an **untested hypothesis**, not a demonstrated mechanism.
- **Scope decision:** exploratory cross-corpus zero-shot experiments from `wa_quick_experiments.ipynb` are outside the official report and planned slides, including backup slides. They may be mentioned informally only if directly asked, clearly labeled exploratory.

## Communication choices

- Organize the report and presentation around the advisor's **three research questions**: (1) prediction and length, (2) incremental rhetorical signal, (3) which strategies and cross-dataset stability.
- Present the **advisor's ten-part narrative**; ten sections do not require exactly ten physical slides.
- Explain CMV and delta quickly; the audience understands ML but may not know the forum.
- Give only a compact overview of all eight strategies. Focus speaking time on **refutation** and **justification** where the results require it.
- One main claim per slide; label every plot with dataset, metric, split/sample and whether numbers are mean±SD.
- A negative or partially replicated result is a valid research finding; report it plainly.
- Do not invent citations, numerical results, significance, or experiments. Verify against the canonical `final-project` branch, especially notebooks in `this work`.
- Keep journal-paper plans separate from the university defense and final-report scope until priorities change.

## Examiner questions worth rehearsing

1. How can strategy scores add value if comment length is such a strong predictor?
2. Does length matching rule out all alternative explanations?
3. Why PR-AUC rather than accuracy or ROC-AUC?
4. How reliable are LLM strategy scores, especially refutation and justification?
5. What exactly was refitted on WA? What was held fixed?
6. Why did the dominant strategy change between datasets? What is tested vs hypothesized?
7. How sensitive are results to column order, sampling, and train/test splits?
8. What is the distinction between predicting delta awards and causing persuasion?

**Maintenance:** Add verified clarifications as needed; if a statement conflicts with a notebook, flag it for verification and never edit research results. Update `PROGRESS.md` separately.
