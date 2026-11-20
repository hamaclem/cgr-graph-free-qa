import argparse
import json
import os
import pandas as pd
import re
from collections import Counter

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
ALIASES_PATH = "data/processed/answer_aliases.json"

def normalize_answer(text):
    text = str(text).casefold()
    text = re.sub(r"\([^)]*\)", "", text)  # Remove parentheses and contents
    text = re.sub(r"[-–—]", " ", text)     # Replace hyphens and dashes
    text = re.sub(r"[^\w\s]", "", text)    # Remove remaining punctuation
    return " ".join(text.split())

def exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

with open(ALIASES_PATH) as f:
    ALIASES = {normalize_answer(ground_truth): {normalize_answer(alias) for alias in aliases} for ground_truth, aliases in json.load(f).items()}

def lenient_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth) or normalize_answer(prediction) in ALIASES.get(normalize_answer(ground_truth), ()))

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        choices=["hu", "qwen2.5", "llama3.1"],
        type=str,
        default="hu"
    )
    parser.add_argument(
        "--embed",
        choices=["medcpt", "hu", "oracle"],     # oracle = gold-article chunks
        type=str,
        default="medcpt"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10
    )
    parser.add_argument(
        "--normalized",
        action=argparse.BooleanOptionalAction,
        default=None  # None = per-model default: hu normalized (cosine), medcpt not (dot product)
    )
    parser.add_argument(
        "--tuples_k",
        type=int,
        default=0
    )
    parser.add_argument(
        "--context",
        choices=["docs", "tuples", "both"],
        type=str,
        default=None  # None = "both" if --tuples_k > 0 else "docs"; must match generate_answers.py
    )

    args = parser.parse_args()

    if args.context is None:
        args.context = "both" if args.tuples_k else "docs"
    if args.context != "docs" and not args.tuples_k:
        parser.error(f"--context {args.context} requires --tuples_k > 0")

    if args.normalized is None:
        args.normalized = args.embed == "hu"

    return args.llm, args.embed, args.top_k, args.normalized, args.tuples_k, args.context


def main():
    llm, embed, top_k, normalized, tuples_k, context = parse_args()

    infix = "" if normalized == (embed == "hu") else ("_normalized" if normalized else "_unnormalized")  # suffix only when deviating from per-model default
    tuples_suffix = f"_tuples{tuples_k}" if tuples_k else ""
    name = f"{llm}_{embed}{infix}_tuplesonly{tuples_k}" if context == "tuples" else f"{llm}_{embed}{infix}_top{top_k}{tuples_suffix}"
    rag_answers_path = f"results/answers/{llm}/answers_{name}.xlsx"
    output_path = f"results/evaluation/answers/{llm}/eval_rag_{name}.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    results_df["em_lenient"] = results_df.apply(
        lambda row: lenient_match(row["a"], row["ground_truth"]),
        axis=1
    )

    metrics_df = pd.DataFrame([{
        "EM": results_df["em"].mean(),
        "EM_lenient": results_df["em_lenient"].mean(),
        "F1": results_df["f1"].mean()
    }])

    print(f"EM: {metrics_df["EM"].iloc[0]:.2%}")
    print(f"EM (lenient): {metrics_df["EM_lenient"].iloc[0]:.2%}")
    print(f"F1: {metrics_df["F1"].iloc[0]:.2%}")

    results_df = results_df[["id", "q", "ground_truth", "a", "em", "em_lenient", "f1"]]

    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name="Answers", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    main()