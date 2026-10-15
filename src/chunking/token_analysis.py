import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

DOCS_PATH = "data/processed/chunks_simple.json"
STATS_PATH = "results/evaluation/chunking/token_statistics_simple.csv"
HISTOGRAM_PATH = "results/evaluation/chunking/token_histogram_simple.png"

def get_token_statistics(tokenizer, docs):
    lengths = []

    for doc in docs:
        tokens = tokenizer(
            doc["text"],
            add_special_tokens=False
        )["input_ids"]

        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths)

    stats = {
        "smallest_section": min(lengths),
        "largest_section": max(lengths),
        "average_tokens": sum(lengths) / n,
        "median_tokens": (
            (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2
            if n % 2 == 0
            else lengths_sorted[n // 2]
        )
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(STATS_PATH, index=False)

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)

    plt.xlabel("Number of tokens")
    plt.ylabel("Number of sections")
    plt.title("Distribution of section sizes")

    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH)

def main():
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)

    get_token_statistics(tokenizer, docs)

if __name__ == "__main__":
   main()