**Model inputs - important methodological distinction**

The main engineered-feature model uses features extracted from both the reply and the original post (OP), including relational features such as differences or matches in sentiment, tone, and rhetorical appeals.

In contrast, the TF-IDF and word-count baselines use reply-only information.

**Potential defense question:** Is the comparison between these models fair?

**Answer:** The models are evaluated on the same held-out comments and threads, but they do not use identical information sources. This is a comparison of predictive approaches, not a controlled experiment isolating the effect of feature complexity. To isolate the contribution of OP-related features, a separate ablation would be required; this was not performed.