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
1. `previous work/` - articles that inspired us
2. `data/` - scripts used to collect the data
3. `this work/` - notebooks with the actual work
4. `results/` - processed datasets produced by this work

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_collection.ipynb` | Parses raw Reddit JSON files and builds the base comments dataset |
| `02_feature_engineering.ipynb` | Adds all text features (sentiment, tone, style, readability, embeddings) |
| `03_model_training.ipynb` | Original model training and evaluation (kept as reference baseline) |

## Raw Data

The raw Reddit JSON files are not stored in this repository due to their size.
They are available on Google Drive (read-only, no sign-in required):

**[Google Drive — raw data folder](https://drive.google.com/drive/folders/1HGik-4OXg2KFpvxPzaGUH625BrYynzmZ?usp=sharing)**

Download the contents and place them under `data/data/` following this structure:
```
data/data/
├── Study2/
│   └── 20180815182030_posts.json
├── chaya/
│   └── combined.json        ← from the redit_data/ folder in Drive
└── downloaded_data/
    └── coded/
        └── 20250210005631_posts.zip
```

> If you only want to re-run the analysis (not re-collect data), you can skip
> `01_data_collection.ipynb` and `02_feature_engineering.ipynb` entirely —
> the processed dataset is already available in `results/cmv_comments_df.csv.zip`.