"""Shared across the 03.x feature-engineering notebooks.

Only plumbing lives here (retries, checkpointed iteration, model loading).
Content that defines what's actually being measured — LLM prompts, the
evidence keyword list — lives here too as plain constants so it isn't
duplicated, but each notebook that uses it should print it inline rather
than just importing it silently, so it stays visible without opening this
file.
"""

import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

# --- Gemini setup -----------------------------------------------------

# 'gemini-flash-latest' is a rolling alias for Google's current-gen flash model — as of 2026-07-14
# it resolves to 'gemini-3.5-flash' (checked via response.model_version). The project originally
# pinned 'gemini-2.5-flash', which is no longer reachable from new API keys/projects (404 NOT_FOUND
# as of 2026-07-14). To revert to that exact older model on a key that still has access, set
# DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash'.
DEFAULT_GEMINI_MODEL = 'gemini-flash-latest'


def make_gemini_client(env_path='../.env'):
    load_dotenv(Path(env_path))  # .env file must contain GEMINI_API_KEY
    api_key = os.environ.get('GEMINI_API_KEY')
    return genai.Client(api_key=api_key)


def generate_content_with_retry(client, model, contents, max_retries=5, initial_delay=2):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except genai_errors.ServerError:
            if attempt == max_retries - 1:
                raise
            print(f"Gemini overloaded (503), retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


class BlockedResponseError(Exception):
    pass


# --- LLM prompts (content, not plumbing — print these in the notebook where used) --

SENTIMENT_LLM_PROMPT_PREFIX = """Rate the sentiment of each of the following numbered texts on a scale from -1 (very negative) to 1 (very positive).

Base your score only on the tone and attitude of the person writing (e.g. hostile, dismissive, or rude vs. friendly, respectful, or warm) - NOT on how negative or positive the topic being discussed is. A calm, respectful discussion of a dark or violent topic should score close to neutral, not negative.

Reply with exactly one line per text, in the format "N: score" (matching the numbers below), and nothing else.

Texts:
"""

TONE_LLM_PROMPT_PREFIX = """Analyze the tone of each of the following numbered texts and classify it as one of the following: Logical, Passionate, Humorous, Sarcastic, Neutral, or Other.

Reply with exactly one line per text, in the format "N: label" (matching the numbers below), and nothing else.

Texts:
"""

STYLE_LLM_PROMPT_PREFIX = """Analyze the writing register of each of the following numbered texts and classify it as one of the following: Formal, Informal.

Reply with exactly one line per text, in the format "N: label" (matching the numbers below), and nothing else.

Texts:
"""

RHETORICAL_LLM_PROMPT_PREFIX = """Rate each of the following numbered texts on three rhetorical appeals, each on a scale from 0 (not present) to 1 (strongly present):

- ethos: appeals to the author's credibility, authority, expertise, or character
- pathos: appeals to emotion (sympathy, anger, fear, hope, etc.)
- logos: appeals to logic and reasoning (facts, statistics, cause-and-effect argument)

These are not mutually exclusive - a text can score high on more than one, or low on all three.

Reply with exactly one line per text, in the format "N: ethos,pathos,logos" (matching the numbers below), and nothing else.

Texts:
"""


# --- checkpointed feature calculation ----------------------------------

def do_calc_feature(df, feature, func, remove_urls=False, switch_order=False):
    for index, row in df.iterrows():
        if pd.isna(row[feature]):  # calculate only if wasn't calculated yet
            print(f"Calculating {feature} for index {index}")
            if not isinstance(row['thread_text'], str):
                text = row['final_comment']
            elif switch_order:
                text = row['final_comment'] + "\n" + row['thread_text']
            else:
                text = row['thread_text'] + "\n" + row['final_comment']

            if remove_urls:
                text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
            df.at[index, feature] = func(text)


def do_calc_feature_op(df, feature, func, remove_urls=False):
    for index, row in df.iterrows():
        if pd.isna(row[feature]):  # calculate only if wasn't calculated yet
            print(f"Calculating {feature} for index {index}")
            text = row['original_post']

            if remove_urls:
                text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
            df.at[index, feature] = func(text)


# --- adjective/adverb ratio --------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load('en_core_web_sm')
    return _nlp


def get_adjective_adverb_ratio(text):
    doc = _get_nlp()(text)
    adjectives = [token.text for token in doc if token.pos_ == 'ADJ']
    adverbs = [token.text for token in doc if token.pos_ == 'ADV']
    return len(adjectives) / len(adverbs) if len(adverbs) > 0 else np.inf


# --- evidence count ------------------------------------------------------

EVIDENCE_URL_EXTENSIONS = ['http://', 'https://', '.com', '.org', '.gov', 'pdf', '.net', 'www.']

EVIDENCE_STEMS = ['data', 'stat', 'statist', 'figur', '%', 'percent', 'averag', 'number', 'amount',
                  'thousand', 'million', 'billion', '$', '€', '¥', '£', 'dollar', 'evid', 'info',
                  'testimoni', 'conclus', 'document', 'experi', 'experi', 'measur', 'measur', 'report',
                  'result', 'census', 'figur', 'plot', 'graph', 'sum', 'total', 'decim', 'digit', 'fraction',
                  'numer', 'half', 'share', 'proport', 'capit', 'cash', 'properti', 'salari', 'wage', 'wealth',
                  'financ', 'resourc', 'roll', 'treasuri', 'bank', 'deposit', 'exchang', 'safe', 'estim', 'price',
                  'price', 'merchandis', 'retail', 'sale', 'cartel', 'invest', 'market', 'deposit', 'document',
                  'indic', 'wit', 'affirm', 'corrobor', 'declar', 'good', 'ground', 'token', 'signific', 'probabl',
                  'p valu', '<', '=', 'greater', 'equal', 'less', 'rang', 'devi', 'sd', 'mode',
                  'median']

_snowball_stemmer = None


def _get_stemmer():
    global _snowball_stemmer
    if _snowball_stemmer is None:
        import nltk
        _snowball_stemmer = nltk.stem.SnowballStemmer('english')
    return _snowball_stemmer


def get_evidence_count(text):
    evidence_count = 0
    comment_stemmed = _get_stemmer().stem(text.lower())

    for word in comment_stemmed.split():
        if any(s == word for s in EVIDENCE_STEMS) and not any(e in word for e in EVIDENCE_URL_EXTENSIONS):
            evidence_count += 1

    return evidence_count


# --- plotting ------------------------------------------------------------

def _autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


def _autolabel_rate(ax, rects, ns):
    for rect, n in zip(rects, ns):
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%\n(n={n})',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


def _draw_counts_panel(ax, labels, convincing_counts, non_convincing_counts, xlabel, title):
    x = range(len(labels))
    width = 0.4
    rects1 = ax.bar([i - width / 2 for i in x], convincing_counts, width, label='Convincing', color='green')
    rects2 = ax.bar([i + width / 2 for i in x], non_convincing_counts, width, label='Non-Convincing', color='red')
    _autolabel(ax, rects1)
    _autolabel(ax, rects2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Comments')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()


def _draw_rate_panel(ax, labels, rates, ns, xlabel, title, baseline_rate):
    x = range(len(labels))
    rects = ax.bar(x, rates, width=0.6, color='steelblue')
    _autolabel_rate(ax, rects, ns)

    ax.axhline(baseline_rate, color='gray', linestyle='--', linewidth=1, label=f'Baseline ({baseline_rate:.1f}%)')
    ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Convincing Rate (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, max(list(rates) + [baseline_rate]) * 1.25)


def _plot_convincing(labels, convincing_counts, non_convincing_counts, rates, ns,
                      xlabel, count_title, rate_title, baseline_rate, figsize, show_counts, panel_gap):
    if show_counts:
        # each panel gets the same size as the standalone rate chart below, side by side
        fig, (ax_counts, ax_rate) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        _draw_counts_panel(ax_counts, labels, convincing_counts, non_convincing_counts, xlabel, count_title)
        _draw_rate_panel(ax_rate, labels, rates, ns, xlabel, rate_title, baseline_rate)
        plt.tight_layout()
        fig.subplots_adjust(wspace=panel_gap)  # applied after tight_layout, which would otherwise collapse the gap
    else:
        fig, ax_rate = plt.subplots(figsize=figsize)
        _draw_rate_panel(ax_rate, labels, rates, ns, xlabel, rate_title, baseline_rate)
        plt.tight_layout()

    plt.show()


def _counts_by_group(df, group_key, index):
    counts = df.groupby([group_key, 'is_convincing'], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(index, fill_value=0)
    convincing_counts = counts[1] if 1 in counts.columns else pd.Series(0, index=index)
    non_convincing_counts = counts[0] if 0 in counts.columns else pd.Series(0, index=index)
    return convincing_counts, non_convincing_counts


def plot_feature(df, feature_name, feature_disp_name, bin_width, figsize=(7, 4), show_counts=False, panel_gap=0.3):
    min_val = df[feature_name].min()
    max_val = df[feature_name].max()

    bins = [min_val + i * bin_width for i in range(int((max_val - min_val) / bin_width) + 2)]
    bin_labels = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(len(bins) - 1)]

    binned = pd.cut(df[feature_name], bins, labels=bin_labels, right=False)
    grouped = df.groupby(binned, observed=True)['is_convincing']
    rates = (grouped.mean() * 100).reindex(bin_labels, fill_value=0)
    ns = grouped.count().reindex(bin_labels, fill_value=0)
    convincing_counts, non_convincing_counts = _counts_by_group(df, binned, bin_labels)

    _plot_convincing(
        bin_labels, convincing_counts.values, non_convincing_counts.values,
        rates.values, ns.values,
        xlabel=f'{feature_disp_name} Bin',
        count_title=f'Number of Convincing and Non-Convincing Comments per {feature_disp_name} Bin',
        rate_title=f'Convincing Rate by {feature_disp_name} Bin',
        baseline_rate=df['is_convincing'].mean() * 100,
        figsize=figsize,
        show_counts=show_counts,
        panel_gap=panel_gap,
    )


def plot_category_histogram(df, column_name, display_name, figsize=(6, 4), show_counts=False, panel_gap=0.3):
    rate_table = df.groupby(column_name, observed=True)['is_convincing'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    labels = rate_table.index.astype(str)
    convincing_counts, non_convincing_counts = _counts_by_group(df, column_name, rate_table.index)

    _plot_convincing(
        labels, convincing_counts.values, non_convincing_counts.values,
        rate_table['mean'].values * 100, rate_table['count'].values,
        xlabel=display_name,
        count_title=f'Number of Convincing and Non-Convincing Comments per {display_name}',
        rate_title=f'Convincing Rate by {display_name}',
        baseline_rate=df['is_convincing'].mean() * 100,
        figsize=figsize,
        show_counts=show_counts,
        panel_gap=panel_gap,
    )
