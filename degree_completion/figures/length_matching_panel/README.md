# Length-matching figure

Two-panel figure for the MSc presentation, illustrating the length-matching
methodology and its effect on the predictive value of rhetorical-strategy
features.

- `make_length_matching_figure.py` — generates both output files below.
- `length_matching_strategy_gain.png` (300 dpi, 16:9) and `.svg`

Regenerate with:

```
source ../../../.venv/bin/activate
python make_length_matching_figure.py
```

## What the figure shows

**Panel A — length matching.** Comment word-count distributions by outcome
class (No Δ vs. Δ Awarded), before and after length matching. "Before" is the
2,753-row scored population; "after" is that same population's comments after
decile-bin matching (point-biserial r drops from 0.288 to 0.061 — the
visually similar before/after histogram shapes are the point).

**Panel B — predictive gain.** Incremental PR-AUC from adding the 8
rhetorical-strategy scores to the engineered-feature model, full scored
population (+0.0664) vs. length-matched sample (+0.0557) — 84% retained.

## Verified matching procedure

Source: `this work/06_strategy_ablation.ipynb`, cells 2, 32, 34–36 (inspected
directly on the `final-project` branch; all numbers below were read from that
notebook's own saved cell outputs, not re-derived).

1. **Population**: `results/05_cmv_comments_df.csv.zip` — the 2,753 comments
   that `05_rhetorical_scoring.ipynb` actually scored on the 8 rhetorical
   strategies (comments below a ~30-word floor and not convincing were
   excluded from scoring upstream, so this is already a subset of the
   project's full 3,971-comment dataset). `word_count` is a pre-existing
   column in this file, not recomputed.
2. **Matching (06, cell 32)**: convincing (`is_convincing == 1`) comments'
   `word_count` is split into 10 quantile bins (`pd.qcut`). Non-convincing
   comments are then binned on the same edges, and for each bin a random
   sample of non-convincing comments equal to that bin's convincing count is
   drawn (`pandas.DataFrame.sample(random_state=42)`). All convincing
   comments are kept; only non-convincing comments are subsampled. This
   yields **1,235 comments** (641 convincing / 594 non-convincing, 51.9%
   positive rate) with point-biserial r(word_count, is_convincing) = 0.061,
   down from r = 0.288 in the unmatched 2,753-row population.
   `make_length_matching_figure.py` reproduces this exact procedure
   (identical code, same `random_state=42`) against the saved results file to
   draw Panel A's "after matching" histogram — it is not a separate,
   re-derived sample.
3. **PR-AUC gain (06, cells 34–35)**: on each of the two samples (full
   population, length-matched sample), two order-robust models (mean over 8
   column-order shuffles, fixed XGBoost+SMOTE architecture) are fit on the
   `trusted` engineered-feature set with and without the 8 strategy-score
   columns added. The reported gain is `PR-AUC(with strategy) −
   PR-AUC(without strategy)` for each sample: **+0.0664** (full, n=2,753) vs.
   **+0.0557** (length-matched, n=1,235) → 0.0557 / 0.0664 = **83.9%**,
   reported as "~84%". These four PR-AUC values are copied directly from the
   notebook's cell 35 output (`length_benchmark_df`) — this script does not
   retrain any model.

## Data sources

| Panel | Data file | Notebook cell(s) |
|---|---|---|
| A (before) | `results/05_cmv_comments_df.csv.zip` | `06_strategy_ablation.ipynb` cell 2 |
| A (after) | same file, re-matched by this script | `06_strategy_ablation.ipynb` cell 32 (reproduced) |
| B | four literal PR-AUC values below | `06_strategy_ablation.ipynb` cells 34–35 |

## Important caveat

**This is not the same length-matched sample as
`04.3_length_controlled_analysis.ipynb`.** That notebook builds its own
strict 1:1 match from the full 3,971-row dataset (also landing at n=1,235 by
coincidence of the same matching parameters, not the same rows) and reports a
different question (collapse of `word_count`/engineered-pipeline/TF-IDF lift
under matching). This figure uses `06`'s own re-match, restricted to the
2,753-row *scored* population, because that is the population the
rhetorical-strategy PR-AUC comparison in Panel B is actually computed on —
`04.3`'s sample includes comments below the scoring floor that have no
strategy score and can't be used for this comparison. `06`'s own markdown
cells flag this distinction explicitly.

As stated on the figure: **descriptive comparison across different samples;
not proof of independence from length.**

## Constraints honored

No notebook, presentation, or existing result file was modified. No model was
retrained; the only computation performed by `make_length_matching_figure.py`
is the row-selection/matching logic already defined in `06`'s cell 32,
re-run read-only against the saved results CSV to extract word-count values
for plotting. All PR-AUC figures are copied literally from verified notebook
output, not recomputed.
