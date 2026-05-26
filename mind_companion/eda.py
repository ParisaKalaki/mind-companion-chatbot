"""
eda.py
------
Exploratory data analysis figures for the report.

Generates 7 PNG figures into reports/figures/:
    fig1_class_distribution.png   — crisis vs non-crisis bar chart
    fig2_text_length.png          — message length distributions overlaid
    fig3_wordcloud_crisis.png     — most common words in crisis posts
    fig4_wordcloud_non_crisis.png — most common words in non-crisis posts
    fig5_counsel_topics.png       — top 15 Counsel Chat topics
    fig6_answer_length.png        — therapist answer length distribution
    fig7_knowledge_base.png       — KB entries grouped by category

Run from anywhere:
    python eda.py
    poetry run python mind_companion/eda.py
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

from config import EXTERNAL_DATA_DIR, FIGURES_DIR, PROCESSED_DATA_DIR


# ---------------------------------------------------------------------------
# Paths (resolved from config.py — independent of cwd)
# ---------------------------------------------------------------------------
CRISIS_TRAIN_PATH = PROCESSED_DATA_DIR / "crisis_train.csv"
COUNSEL_PATH      = PROCESSED_DATA_DIR / "counsel_chat_clean.csv"
KB_PATH           = EXTERNAL_DATA_DIR / "knowledge_base.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load crisis training data
# ---------------------------------------------------------------------------
df = pd.read_csv(CRISIS_TRAIN_PATH)


# ---------------------------------------------------------------------------
# FIGURE 1: Class Distribution
# ---------------------------------------------------------------------------
counts = df['label'].value_counts()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ['Non-Crisis', 'Crisis'],
    [counts['non-suicide'], counts['suicide']],
    color=['#10b981', '#ef4444'],
    edgecolor='white',
    width=0.5,
)
for bar, count in zip(bars, [counts['non-suicide'], counts['suicide']]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f'{count:,}',
        ha='center', fontsize=12, fontweight='bold',
    )
ax.set_title('Class Distribution: Crisis vs Non-Crisis Posts',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Number of Posts', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylim(0, 95000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_class_distribution.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# FIGURE 2: Text Length Distribution
# ---------------------------------------------------------------------------
df['text_length'] = df['text'].astype(str).apply(len)
crisis     = df[df['label'] == 'suicide']['text_length']
non_crisis = df[df['label'] == 'non-suicide']['text_length']

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(non_crisis.clip(upper=2000), bins=60,
        alpha=0.6, color='#10b981', label='Non-Crisis', edgecolor='none')
ax.hist(crisis.clip(upper=2000), bins=60,
        alpha=0.6, color='#ef4444', label='Crisis', edgecolor='none')
ax.axvline(crisis.median(), color='#ef4444', linestyle='--',
           linewidth=2, label=f'Crisis median: {int(crisis.median())}')
ax.axvline(non_crisis.median(), color='#10b981', linestyle='--',
           linewidth=2, label=f'Non-Crisis median: {int(non_crisis.median())}')
ax.set_title('Text Length Distribution: Crisis vs Non-Crisis',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Message Length (characters)', fontsize=12)
ax.set_ylabel('Number of Posts', fontsize=12)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_text_length.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# FIGURE 3: Word Cloud — Crisis Posts
# ---------------------------------------------------------------------------
stopwords = set(STOPWORDS)
stopwords.update([
    'will', 'one', 'now', 'just', 'like',
    'know', 'really', 'want', 'feel', 'get',
    'go', 'im', 'ive', 'dont', 'cant',
])

crisis_text = ' '.join(df[df['label'] == 'suicide']['text'].astype(str).tolist())
wc = WordCloud(
    width=1000, height=500,
    background_color='black',
    colormap='Reds',
    stopwords=stopwords,
    max_words=100,
    collocations=False,
).generate(crisis_text)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('Most Common Words: Crisis Posts',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_wordcloud_crisis.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# FIGURE 4: Word Cloud — Non-Crisis Posts
# ---------------------------------------------------------------------------
non_crisis_text = ' '.join(
    df[df['label'] == 'non-suicide']['text'].astype(str).tolist()
)
wc2 = WordCloud(
    width=1000, height=500,
    background_color='black',
    colormap='Greens',
    stopwords=stopwords,
    max_words=100,
    collocations=False,
).generate(non_crisis_text)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc2, interpolation='bilinear')
ax.axis('off')
ax.set_title('Most Common Words: Non-Crisis Posts',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_wordcloud_non_crisis.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# FIGURE 5: Counsel Chat Topic Distribution
# ---------------------------------------------------------------------------
df2 = pd.read_csv(COUNSEL_PATH)
print(df2.columns.tolist())
print(df2.head(2))

topic_counts = df2['topic'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    topic_counts.index[::-1],
    topic_counts.values[::-1],
    color='#10b981',
    edgecolor='white',
)
for bar, val in zip(bars, topic_counts.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=9)
ax.set_title('Top 15 Topics: Counsel Chat Dataset',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Number of Questions', fontsize=12)
ax.set_ylabel('Topic', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig5_counsel_topics.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# FIGURE 6: Therapist Answer Length
# ---------------------------------------------------------------------------
df2['answer_length'] = df2['answerText'].astype(str).apply(len)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df2['answer_length'].clip(upper=3000), bins=50,
        color='#10b981', edgecolor='white', alpha=0.85)
ax.axvline(df2['answer_length'].median(), color='#ef4444',
           linestyle='--', linewidth=2,
           label=f"Median: {int(df2['answer_length'].median())} chars")
ax.set_title('Therapist Answer Length: Counsel Chat Dataset',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Answer Length (characters)', fontsize=12)
ax.set_ylabel('Number of Answers', fontsize=12)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig6_answer_length.png', dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# FIGURE 7: Knowledge Base Category Breakdown
# ---------------------------------------------------------------------------
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

categories = {}
for entry in kb:
    cat = entry.get("category", "Unknown")
    categories[cat] = categories.get(cat, 0) + 1

cat_df = pd.Series(categories).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(cat_df.index, cat_df.values,
               color='#6366f1', edgecolor='white', alpha=0.9)
for bar, val in zip(bars, cat_df.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=10, fontweight='bold')
ax.set_title('Knowledge Base: Entries by Category',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Number of Entries', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig7_knowledge_base.png', dpi=150)
plt.close()

print(f"\n✅ Generated 7 figures in {FIGURES_DIR}")