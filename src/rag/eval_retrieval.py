import os
import argparse
import pandas as pd
from urllib.parse import urlparse, unquote

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
TOP_K = {"chunks": [1, 3, 5, 10], "tuples": [10, 30, 50, 100]}

REDIRECTS = {"AIDS": "Management_of_HIV/AIDS", "perfusion_scan": "Ventilation/perfusion_scan"}

def extract_article(article):
    article = unquote(urlparse(article).path.split("/wiki/", 1)[-1])

    return REDIRECTS.get(article, article)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",  
        choices=["medcpt", "hu"],
        default="medcpt"
    )
    parser.add_argument(
        "--normalized",
        action=argparse.BooleanOptionalAction,
        default=None  # None = per-model default: hu normalized (cosine), medcpt not (dot product)
    )
    parser.add_argument(
        "--source", 
        choices=["chunks", "tuples"], 
        default="chunks"
    )
    args = parser.parse_args()
    if args.normalized is None:
        args.normalized = args.embed == "hu"

    suffix = "" if args.normalized == (args.embed == "hu") else ("_normalized" if args.normalized else "_unnormalized")  # suffix only when deviating from per-model default
    suffix += "_tuples" if args.source == "tuples" else ""

    results_path = f"results/retrieval/retrieval_results_{args.embed}{suffix}.xlsx"
    output_path = f"results/evaluation/retrieval/eval_retrieval_{args.embed}{suffix}.xlsx"

    ground_truth = pd.read_excel(GROUND_TRUTH_PATH)
    retrieval_results = pd.read_excel(results_path)

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
        ].sort_values("rank")

        result = {
            "query": query,
            "relevant_articles": ", ".join(sorted(articles))
        }

        for k in TOP_K[args.source]:
            top_k_articles = set(retrieved["article"].head(k))
            found = articles.intersection(top_k_articles)

            recall = len(found) / 2
            result[f"retrieved_articles_at_{k}"] = ", ".join(sorted(top_k_articles))
            result[f"found_articles_at_{k}"] = ", ".join(sorted(found)) 
            result[f"recall_at_{k}"] = recall

        evaluation.append(result)

    evaluation_df = pd.DataFrame(evaluation)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        evaluation_df.to_excel(writer, sheet_name="Articles", index=False)

        recalls = pd.DataFrame([{
            f"mean_recall_at_{k}": evaluation_df[f"recall_at_{k}"].mean()
            for k in TOP_K[args.source]
        }])
    
        recalls.to_excel(writer, sheet_name="Recall", index=False)
if __name__ == "__main__":
    main()