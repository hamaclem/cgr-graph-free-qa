import pandas as pd
from urllib.parse import urlparse, unquote

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
RESULTS_PATH = "results/retrieval/retrieval_results.xlsx"
OUTPUT_PATH = "results/evaluation/retrieval/eval_retrieval.csv"

def extract_article(article):
    article = urlparse(article).path.split("/")[-1]
    article = unquote(article)

    return article

def main():
    ground_truth = pd.read_excel(GROUND_TRUTH_PATH)
    retrieval_results = pd.read_excel(RESULTS_PATH)

    relevant_articles = {}

    for _, row in ground_truth.iterrows():
        query = row["q"]

        articles = {
            extract_article(row["wiki1"]),
            extract_article(row["wiki2"])
        }

        relevant_articles[query] = articles

    evaluation = []

    for query, articles in relevant_articles.items():
        retrieved = retrieval_results[
            retrieval_results["query"] == query
        ]

        retrieved_articles = list(dict.fromkeys(retrieved["article"]))
        retrieved_unique = set(retrieved_articles)
        found = articles.intersection(retrieved_unique)

        recall = len(found) / 2

        evaluation.append({
            "query": query,
            "relevant_articles": ", ".join(sorted(articles)),
            "retrieved_articles": ", ".join(retrieved_articles),
            "found_articles": ", ".join(sorted(found)),
            "relevant_found": len(found),
            "recall_at_5": recall
        })

    evaluation_df = pd.DataFrame(evaluation)

    evaluation_df.to_csv(OUTPUT_PATH, index=False)

    summary = pd.DataFrame([{
        "mean_recall_at_5": evaluation_df["recall_at_5"].mean(),
        "both_found": (evaluation_df["recall_at_5"] == 1.0).mean(),
        "one_found": (evaluation_df["recall_at_5"] == 0.5).mean(),
        "none_found": (evaluation_df["recall_at_5"] == 0.0).mean()
    }])

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()