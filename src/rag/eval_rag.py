import pandas as pd
import re
from collections import Counter

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
RAG_ANSWERS_PATH = "results/answers/rag_answers.xlsx"
OUTPUT_PATH = "results/evaluation/eval_results.xlsx"

def normalize_answer(text):
    text = str(text).casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def token_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if not pred_tokens or num_same == 0:
        return 0.0

    prec = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * prec * recall / (prec + recall)

def main():
    ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)
    rag_df = pd.read_excel(RAG_ANSWERS_PATH)

    ground_truth_df = ground_truth_df.rename(
        columns={"a": "ground_truth"}
    )

    results_df = rag_df.merge(
        ground_truth_df[["q", "ground_truth"]],
        on="q",
    )

    results_df["em"] = results_df.apply(
        lambda row: exact_match(row["a"], row["ground_truth"]),
        axis=1
    )

    results_df["f1"] = results_df.apply(
        lambda row: token_f1(row["a"], row["ground_truth"]),
        axis=1
    )

    print(f"EM: {results_df["em"].mean():.2%}")
    print(f"F1: {results_df["f1"].mean():.2%}")

    results_df = results_df[["q", "a", "ground_truth", "em", "f1"]]
    results_df.to_excel(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()