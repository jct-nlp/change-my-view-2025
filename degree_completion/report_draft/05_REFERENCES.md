# 5. References

## 5.1 Cited literature

- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

- Labruna, T., Modzelewski, A., Satta, G., & Da San Martino, G. (2026). Detecting winning arguments with large language models and persuasion strategies. In *Findings of the Association for Computational Linguistics: EACL 2026* (pp. 1888–1915). Association for Computational Linguistics. https://doi.org/10.18653/v1/2026.findings-eacl.97

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

- Na, R. W., & DeDeo, S. (2022). The diversity of argument-making in the wild: From assumptions and definitions to causation and anecdote in Reddit's "Change My View." In J. Culbertson, A. Perfors, H. Rabagliati, & V. Ramenzoni (Eds.), *Proceedings of the 44th Annual Meeting of the Cognitive Science Society*. Cognitive Science Society. https://escholarship.org/uc/item/20n240qq

- Priniski, J. H., & Horne, Z. (2018). Attitude change on Reddit's Change My View. In *Proceedings of the 40th Annual Meeting of the Cognitive Science Society* (pp. 2279–2284). Cognitive Science Society.

- Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. In *Proceedings of the 25th International Conference on World Wide Web* (WWW '16) (pp. 613–624). International World Wide Web Conferences Steering Committee. https://doi.org/10.1145/2872427.2883081

## 5.2 Code reuse

The Reddit data-collection approach (§2.1) and the evidence-marker keyword-stem feature (§2.2) both adapt code released by Priniski and Horne alongside their 2018 study — `collect_posts.py` and `discussion_analysis.py`, respectively — rather than reimplementing either independently. The code is available at https://github.com/jpriniski/CMV.

## 5.3 Systems and libraries

The pipeline draws on PRAW (Reddit API client), Google's `google-genai` SDK (Gemini), HuggingFace Transformers models (a persuasive-language detector, `chreh/persuasive_language_detector`, and a sentence-embedding model, `BAAI/bge-base-en-v1.5`, both retained; a RoBERTa-based emotion classifier and a zero-shot `bart-large-mnli` classifier, both trialed and rejected per §2.2), spaCy, `textstat`, XGBoost, `imbalanced-learn` (SMOTE), SHAP, and ConvoKit (used to access the Winning Arguments corpus for external validation, §2.9).

## 5.4 AI tools used

Google Gemini was used as the LLM underlying several engineered features (tone, formality, and rhetorical-appeal scores) and the rhetorical-strategy scoring rubric described in §2.2–§2.6; its role in the research method is documented throughout the Methods chapter. Claude Code (Anthropic) was used as an AI coding assistant during the development of the data-processing and modeling pipeline and the accompanying notebooks, and also assisted with drafting and editing this report's text. The research design, analysis, and conclusions are the author's own.

## 5.5 Code and Data Availability

The project's notebooks, supporting code (`common_functions.py`), and `README.md` are included directly with this submission. `README.md` documents setup, the notebook execution order, and the project's working data files. The raw Reddit data is not included in the submission due to its size; it is available via the read-only Google Drive link given in `README.md`. The Winning Arguments corpus used for external validation (§2.9) is fetched automatically through ConvoKit and is not distributed as part of this submission.
