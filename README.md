# change-my-view-2025

This work is the final project in Text Mining curse at the Jerusalem College of Technology, 2025.

Change My View (CMV) is a popular subreddit on Reddit where users engage in structured debates with the goal of persuading 
others to reconsider their viewpoints. Unlike typical internet arguments, CMV encourages respectful discourse, requiring 
users to provide well-reasoned arguments and evidence. A core feature of CMV is the "delta" system: if a user successfully 
changes another participant’s mind, the persuaded user awards them a "delta" symbol (∆). This delta serves as an indicator 
of a convincing argument, making CMV a unique platform for studying persuasion in online discussions. The structured format 
of the forum, combined with the delta-based feedback, provides a valuable dataset for analyzing what makes an argument compelling.

Previous research has leveraged CMV discussions to study persuasion, but earlier studies were often limited by the analytical
methods available at the time. With recent advancements in natural language processing (NLP) and machine learning, we now 
have the tools to extract more nuanced insights from text data. By applying modern text mining techniques, such as 
transformer-based language models and sophisticated feature engineering methods, we can potentially improve upon previous 
results and enhance our understanding of what makes an argument compelling. This work has practical applications in areas 
such as automated debate analysis, content moderation, and persuasive writing assistance.

More details can be found in the introduction section in the notebook located under "this work" directory.

This repo contains the following:
1. `previous work/` - articles that inspired this project
2. `data/` - raw and processed data files
3. `this work/` - notebooks with the actual work, plus `common_functions.py` (shared helpers used across several notebooks)
4. `results/` - processed datasets produced by this work

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each time you open a new terminal: `source .venv/bin/activate`

Create a `.env` file in the repo root with:

```
# Reddit API (only needed to re-run 01_data_download.ipynb)
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USERNAME=...
REDDIT_PASSWORD=...

# Gemini API (needed for 03.1, 03.2, 03.3, 05, and 08 - LLM-based feature/strategy scoring)
GEMINI_API_KEY=...
```

Get Reddit credentials from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) and a Gemini key from
[Google AI Studio](https://aistudio.google.com/). Gemini's free tier is capped at 20 requests/day per project
and can be unreliable under load; a billing-enabled key (minimum $5 prepay) removes the daily cap and was
needed to finish this project's larger scoring runs in practical time.

**macOS (Apple Silicon) note:** this project's `.venv` runs on x86_64 Python under Rosetta, to match
precompiled wheels (e.g. XGBoost's `libxgboost.dylib`) that don't ship natively for arm64. Two consequences:
XGBoost needs `libomp` installed via an Intel-prefixed Homebrew (`/usr/local/bin/brew install libomp` -
note the `/usr/local/bin/brew` path itself, not `/opt/homebrew`, is what selects the Intel install; the
same binary is used to `uninstall libomp` afterward if you don't need it beyond this project), and
`convokit` (used by `08`) needs its `bottleneck` dependency installed as a prebuilt wheel rather than
built from source: `pip install --only-binary=:all: bottleneck`, then `pip install --prefer-binary convokit`.

## Notebooks

Run in order for a first pass; each notebook after `02` can also be re-run independently by loading its
predecessor's saved output, without re-running the whole chain.

| Notebook | Description |
|---|---|
| `00_introduction.ipynb` | Project background and motivation (expanded version of this README's intro). |
| `01_data_download.ipynb` | Downloads new Reddit data via PRAW. Requires Reddit credentials in `.env`. Only needed to collect new data - output is timestamped JSON under `data/downloaded_data/`. |
| `02_data_annotation.ipynb` | Parses raw Reddit JSON into one row per comment (original post, thread context, final comment, `is_convincing` label). Output: `results/02_comments_df_base.csv.zip`. |
| `03.1_feature_engineering_part1.ipynb` | Feature engineering, part 1 of 3: sentiment, tone, style. |
| `03.2_feature_engineering_part2.ipynb` | Feature engineering, part 2 of 3: rhetorical means (ethos/pathos/logos, LLM-scored), adjective-adverb ratio, readability, evidence use, persuasive-language classification, document embeddings. |
| `03.3_feature_engineering_op_relation.ipynb` | Feature engineering, part 3 of 3: the same feature families computed on the original post and its relation to the comment. Combines all of `03.1`-`03.3` into the project's main working dataset. Output: `results/03_cmv_comments_df.csv.zip`. |
| `04.1_model_training.ipynb` | Honest evaluation: thread-grouped train/test split (no leakage), PR-AUC as the primary metric, baselines (majority-class, stratified-random, TF-IDF+LogReg), and a quantified comparison against a naive (leaky) split. |
| `04.2_baseline_comparison.ipynb` | Investigates why TF-IDF beat the engineered-feature model in `04.1`: feature-family ablation, TF-IDF interpretability, the `word_count` discovery, and a stacked TF-IDF+engineered ensemble. Adds `word_count` as a feature. |
| `04.3_length_controlled_analysis.ipynb` | Controls for the `word_count` confound found in `04.2`: length-matched resampling, model re-selection on the matched sample, and a content-motivated length floor (≥30 words) adopted as the working dataset. Output: `results/04_cmv_comments_df.csv.zip`. |
| `05_rhetorical_scoring.ipynb` | The project's main contribution: an 8-strategy rhetorical rubric, scored per comment by the Gemini API, with test-retest and hand-scoring validation. Requires `GEMINI_API_KEY`. Output: `results/05_cmv_comments_df.csv.zip`. |
| `06_strategy_ablation.ipynb` | Does adding the strategy scores improve the model? Feature ablation, model re-selection, and a length-controlled benchmark isolating the strategy-score effect from the `word_count` confound. |
| `07_strategy_interpretation.ipynb` | Which strategies drive the result? SHAP beeswarm and Mann-Whitney effect sizes per strategy, on the full population and a length-matched sample. |
| `08_wa_external_validation.ipynb` | External validation on the independent Winning Arguments (WA) corpus (Tan et al. 2016, via `convokit`) - same rubric, prompt, and model architecture as `05`-`07`, unchanged, to check whether the findings generalize beyond this project's own CMV sample. |

## Raw Data

The raw data files are not stored in this repository due to their size.
They are available on Google Drive (read-only, no sign-in required):

**[Google Drive — raw data folder](https://drive.google.com/drive/folders/1HGik-4OXg2KFpvxPzaGUH625BrYynzmZ?usp=sharing)**

Download the contents and place them under `data/` following this structure:

```
data/
├── Study2/
│   └── 20180815182030_posts.json
├── chaya/
│   └── combined.json
└── downloaded_data/
    └── coded/
        └── 20250210005631_posts.zip
```

### Per-file sources

| File | Size | Source |
|------|------|--------|
| `data/Study2/20180815182030_posts.json` | ~52 MB | From the [jpriniski/CMV](https://github.com/jpriniski/CMV/tree/master/Study2) GitHub repo (also mirrored to Drive) |
| `data/chaya/combined.json` | ~329 MB | Generated by running `02_data_annotation.ipynb` from the Drive source files, or download the pre-built version from the `redit_data/` folder in Drive |
| `data/downloaded_data/coded/20250210005631_posts.zip` | ~89 MB | Available in the `coded/` folder in Drive |

> If you only want to re-run the analysis (not re-collect or re-annotate raw data), you can skip
> `01_data_download.ipynb` and `02_data_annotation.ipynb` entirely and start from whichever later
> notebook's input is already available - each notebook above lists its own output file, and every
> notebook from `03.3` onward loads directly from a previous notebook's `results/*.csv.zip`, not from
> raw data.

The Winning Arguments corpus used by `08_wa_external_validation.ipynb` is downloaded automatically via
`convokit` (see the macOS setup note above) and does not need to be fetched from Drive.
