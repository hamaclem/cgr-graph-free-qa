import json
import re
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

ARTICLES_PATH = "data/raw/articles.json"
STATS_PATH = "results/evaluation/chunking/chunking_statistics.csv"
HISTOGRAM_PATH = "results/evaluation/chunking/chunking_histograms.png"

def calculate_article_stats(articles, tokenizer):
    stats = []

    for title, text in articles.items():
        stats.append({
            "title": title,
            "characters": len(text),
            "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "words": len(re.findall(r"\S+", text)),
        })

    return pd.DataFrame(stats)

def calculate_column_stats(column):
    return [
        column.mean(),
        column.median(),
        column.std(),
        column.quantile(0.75),
        column.quantile(0.90),
        column.quantile(0.95),
        column.min(),
        column.max(),
    ]

def calculate_descriptive_stats(df):
    stats = pd.DataFrame({
        "characters": calculate_column_stats(df["characters"]),
        "tokens": calculate_column_stats(df["tokens"]),
        "words": calculate_column_stats(df["words"]),
    })

    stats.index = [
        "mean",
        "median",
        "std",
        "75%",
        "90%",
        "95%",
        "min",
        "max",
    ]
    
    stats.to_csv(STATS_PATH)

    return stats

def create_histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # characters
    axes[0].hist(df["characters"], bins=30)
    axes[0].set_xlabel("Number of characters")
    axes[0].set_ylabel("Number of articles")
    axes[0].set_title("Distribution of characters per article")

    # tokens
    axes[1].hist(df["tokens"], bins=30)
    axes[1].set_xlabel("Number of tokens")
    axes[1].set_ylabel("Number of articles")
    axes[1].set_title("Distribution of tokens per article")

    # words
    axes[2].hist(df["words"], bins=30)
    axes[2].set_xlabel("Number of words")
    axes[2].set_ylabel("Number of articles")
    axes[2].set_title("Distribution of words per article")

    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH)

def main():
    with open(ARTICLES_PATH, "r") as f:
        articles = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    df = calculate_article_stats(articles,tokenizer)
    calculate_descriptive_stats(df)
    create_histograms(df)

if __name__ == "__main__":
   main()