# Appendix A

## Table 2: Feature-Family Summary

Table 2, referenced from §2.2, is reproduced here in full because its five columns do not fit comfortably within a single portrait page. It is split into three parts (Table 2a–2c) so that each fits cleanly on one page.

**Table 2a.** Feature-family summary, part 1 of 3 — sentiment, tone, and style families. With each
family's source notebook and whether it belongs to the trusted or untrusted set. For the two
one-hot-encoded rows here, the category count in the feature-group name is the true number of
labels the classifier assigns; "Columns" is one less per side, since the first category of each
encoding is dropped as an implicit reference level (§2.2).

| **Feature / feature group** | **Variants included** | **Source notebook(s)** | **Columns** | **Set** |
|---|---|---|---|---|
| Sentiment (LLM, tone-focused) | comment, OP, diff | 03.1, 03.3 | 3 | Trusted |
| Sentiment (VADER, lexicon-based) | comment only | 03.1 | 1 | Untrusted |
| Tone (LLM, 6 categories, one-hot) | comment, OP | 03.1, 03.3 | 10 (2 sides × 5, first category dropped per side) | Trusted |
| Tone (RoBERTa emotion classifier, 7 categories, one-hot) | comment only | 03.1 | 6 (1 side × 6, first category dropped) | Untrusted |
| Style / formality (LLM-assigned, binary) | comment, OP | 03.1, 03.3 | 2 | Trusted |
| Style (BART-large-MNLI zero-shot, 2 categories, one-hot) | comment only | 03.1 | 1 (1 side × 1, first category dropped) | Untrusted |

```{=typst}
#pagebreak()
```

**Table 2b.** Feature-family summary, part 2 of 3 — rhetorical appeals, style/readability, and
relational features.

| **Feature / feature group** | **Variants included** | **Source notebook(s)** | **Columns** | **Set** |
|---|---|---|---|---|
| Rhetorical appeals: ethos, pathos, logos (LLM, 0–1 scale) | comment, OP, diff | 03.2, 03.3 | 9 | Trusted |
| Rhetorical appeals: ethos, pathos, logos (Empath lexicon) | comment only | 03.2 | 3 | Untrusted |
| Adjective–adverb ratio | comment, OP, diff | 03.2, 03.3 | 3 | Trusted |
| Readability: Flesch–Kincaid grade level | comment, OP, diff | 03.2, 03.3 | 3 | Trusted |
| Readability: Flesch reading ease | comment only (no OP/diff counterpart — dropped as redundant with `fk_grade_level_op`, r = −0.905) | 03.2 | 1 | Trusted |
| Evidence-marker count | comment, OP, diff | 03.2, 03.3 | 3 | Trusted |
| Comment↔OP match indicators (sentiment, tone, style, ethos, pathos, logos) | boolean agreement, computed from original comment/OP values, not derived from the "diff" columns above (bucketed same/different, not a thresholded difference) | 03.3 | 6 | Trusted |

```{=typst}
#pagebreak()
```

**Table 2c.** Feature-family summary, part 3 of 3 — the two remaining feature families, plus the
representation totals carried forward into model selection (§2.3).

| **Feature / feature group** | **Variants included** | **Source notebook(s)** | **Columns** | **Set** |
|---|---|---|---|---|
| Persuasive-language detector output | comment only | 03.2 | 1 | Trusted |
| Document embeddings (`BAAI/bge-base-en-v1.5`, PCA) | comment only | 03.2 | 187 | Trusted |
| **Handcrafted Trusted Total** | | | **41** | Trusted |
| **Handcrafted Untrusted Total** | | | **11** | Untrusted |
| **Trusted (handcrafted + embeddings)** | | | **228** | — |
| **Trusted + untrusted** | | | **239** | — |
