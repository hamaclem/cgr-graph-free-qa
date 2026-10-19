import argparse
import pandas as pd
import re
from collections import Counter

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="hu")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model = args.llm
    top_k = args.top_k
    rag_answers_path = f"results/answers/rag_answers_{model}_top{top_k}.xlsx"
    output_path = f"results/evaluation/answers/eval_rag_{model}_top{top_k}.xlsx"

    rag_df = pd.read_excel(rag_answers_path)

    ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)
    ground_truth_df = ground_truth_df.rename(
        columns={"a": "ground_truth"}
    )

    results_df = rag_df.merge(
        ground_truth_df[["id", "ground_truth"]],
        on="id",
    )

    results_df["em"] = results_df.apply(
        lambda row: exact_match(row["a"], row["ground_truth"]),
        axis=1
    )

    results_df["f1"] = results_df.apply(
        lambda row: token_f1(row["a"], row["ground_truth"]),
        axis=1
    )

    metrics_df = pd.DataFrame([{
        "EM": results_df["em"].mean(),
        "F1": results_df["f1"].mean()
    }])

    print(f"EM: {metrics_df["EM"].iloc[0]:.2%}")
    print(f"F1: {metrics_df["F1"].iloc[0]:.2%}")

    results_df = results_df[["id", "q", "ground_truth", "a", "em", "f1"]]

    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name="Answers", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    main()