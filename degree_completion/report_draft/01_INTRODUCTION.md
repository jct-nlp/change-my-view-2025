# 1. Introduction

## 1.1 Background and Motivation

Reddit's Change My View (CMV) subreddit provides a research setting for studying persuasion in naturally occurring online discussions (Priniski & Horne, 2018; Na & DeDeo, 2022). In CMV, an original poster (OP) presents a view they are willing to reconsider, and other participants respond with arguments. When an OP considers a response persuasive, they can award it a *delta* (∆). This mechanism provides an observable, comment-level indicator of reported persuasive success, rather than a label assigned solely by an external evaluator.

A delta is nevertheless an imperfect outcome measure. It records the OP's reported response within the norms and incentives of the forum; it does not independently establish a lasting change in belief. Deltas may also be withheld for reasons unrelated to the quality of an argument. Accordingly, this study treats delta-award status as an operational label for persuasive success, not as direct evidence of an underlying psychological change.

Predicting delta awards poses a further interpretive challenge. Successful and unsuccessful comments may differ in superficial properties, especially length, as well as in how their arguments are constructed. A model may achieve useful predictive performance by exploiting such surface-level properties without identifying meaningful rhetorical differences. Understanding whether explicitly defined rhetorical strategies add predictive information beyond comment length is therefore central to interpreting what persuasion-prediction models learn.

## 1.2 Research Problem and Objectives

This project investigates the prediction of delta-award status for CMV comments using textual features. The task is formulated as binary classification at the comment level, with evaluation splits grouped by discussion thread to prevent comments from the same thread appearing in both training and test sets. The analysis examines conventional engineered textual features, comment length, and scores assigned to defined rhetorical strategies.

The investigation evolved in response to an important empirical result: comment length alone outperformed a substantially more elaborate engineered-feature pipeline. This finding sharpened the research question from whether textual properties predict delta awards to **which predictive signals remain after accounting for length**. The subsequent length-controlled comparisons, strategy ablations, and interpretation analyses were developed to address that question. The report describes this progression as it occurred, rather than presenting all later analyses as elements of a fully predetermined design.

The objectives are to characterize the predictive role of textual features and length; assess whether rhetorical-strategy scores provide incremental predictive information; identify which strategies are most informative under different analytical measures; and examine how the findings behave in a second persuasion dataset constructed using a different methodology.

## 1.3 Research Questions

The project is organized around three research questions:

- **RQ1.** To what extent can persuasive success be predicted from textual features, and what role does comment length play?
- **RQ2.** Do rhetorical-strategy measures provide incremental predictive information beyond comment length and other textual features?
- **RQ3.** Which rhetorical strategies contribute to prediction, and how stable are these findings across datasets constructed using different methodologies?

RQ1 establishes the predictive setting and the importance of length. RQ2 examines whether the rhetorical-strategy measures add information to the existing feature representation and how their observed contribution changes under length matching. RQ3 considers the importance of individual strategies and the extent to which their patterns are consistent across datasets.

## 1.4 Research Approach and Contributions

The study constructs a dataset of 3,971 comments from CMV discussions (referred to below as the home dataset), including 641 delta-awarded comments (16.1%). It evaluates an engineered textual-feature representation alongside simpler baselines, using thread-grouped train/test separation. The engineered features include measures of sentiment, tone, rhetorical appeals, readability, and evidence markers, extracted from both the reply and the original post, as well as relational features capturing differences and similarities between them. In contrast, the TF-IDF and word-count baselines use comment-only information. Some features were derived using large language models (LLMs).

The initial comparisons reveal that word count alone achieves higher predictive performance than the engineered-feature pipeline on the home dataset's held-out test set. The project therefore examines differences in comment length by outcome and performs length-controlled comparisons. These analyses show that the predictive advantage of several feature representations diminishes markedly when length differences are reduced. They also motivate a more focused investigation of rhetorical features. Length matching reduces an important confound, but it does not by itself establish causal effects or complete independence from length.

Eight rhetorical strategies relevant to CMV discussions, including refutation, justification, concession, and evidence use, are defined and scored using an LLM-based rubric. This approach adapts a general scoring methodology described by Labruna et al. (2026) to a different taxonomy and setting. The resulting scores are evaluated through ablation experiments and examined using both SHAP-based model interpretation and univariate rank-biserial effect sizes. The addition of strategy scores improves predictive performance over the engineered-feature representation on the evaluated subset of the home dataset; a substantial portion of the observed incremental gain also remains in a separately constructed length-matched analysis. The interpretation of individual strategies depends on the metric: refutation has the highest model-based SHAP importance in the home-dataset analysis, whereas justification has the largest univariate rank-biserial effect size. These measures answer different questions and should not be treated as interchangeable.

The study additionally evaluates the rhetorical-strategy approach on Winning Arguments, a persuasion dataset constructed using a different sampling methodology (Tan et al., 2016). The model architecture and configuration selected on the home dataset were retained, but a new model instance was trained on Winning Arguments' training data and evaluated on its held-out data. This is a transfer of the modeling procedure, not a zero-shot evaluation of the original fitted model. Rhetorical-strategy scores show predictive utility in this second dataset. The relative importance of individual strategies varies across datasets: refutation is the leading rhetorical strategy by SHAP importance in the home dataset, whereas justification leads in Winning Arguments. By contrast, justification has the largest univariate rank-biserial effect size in both datasets. However, because a directly comparable length-controlled ablation was not performed on Winning Arguments, the external evaluation does **not** establish that the incremental contribution of rhetorical strategies beyond length replicates to the same extent as in the home dataset.

The study's contributions are primarily empirical and analytical: (1) a documented examination of the strength of comment length as a predictor of delta awards and of its implications for model interpretation; (2) a systematic assessment of the incremental predictive value of an eight-strategy rhetorical representation, including length-controlled comparisons and a limited scoring-reliability assessment; and (3) an external evaluation that identifies both recurring predictive utility and differences in strategy-level patterns across datasets. The work adapts existing modeling and scoring approaches rather than claiming a new learning algorithm or establishing causal effects of rhetorical strategies.

## 1.5 Report Organization

The remainder of the report describes the relevant prior research and the methods used for dataset construction, feature extraction, model training and evaluation, length-controlled analysis, rhetorical-strategy scoring, and external validation. The Results chapter presents the corresponding quantitative findings. The Discussion interprets those findings in relation to the three research questions and prior work, examines limitations of the labels, scoring process, and evaluation design, and outlines directions for future research. The report concludes by summarizing the principal empirical contributions.

References and information on accessing the project's code and data are provided in the final sections.
