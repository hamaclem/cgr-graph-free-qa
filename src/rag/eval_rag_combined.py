from pathlib import Path
import pandas as pd

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
OUTPUT_PATH = "results/evaluation/answers/eval_rag_combined.xlsx"

def main():
    models = ["hu", "qwen2.5", "llama3.1"]
    top_ks = [5]

    ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)
    combined_df = ground_truth_df[["id", "q", "a"]].rename(
        columns={"a": "ground_truth"}
    )

    metrics = []

    for model in models:
        for top_k in top_ks:
            file = Path(
                f"results/evaluation/answers/eval_rag_{model}_top{top_k}.xlsx"
            )

            df = pd.read_excel(file)

            answers = df[["id", "a"]].rename(columns={"a": f"a_{model}_top{top_k}"})
            combined_df = combined_df.merge(answers, on="id", how="left")

            metrics.append({
                "model": model,
                "top_k": top_k,
                "EM": df["em"].mean(),
                "F1": df["f1"].mean()
            })

    metrics_df = pd.DataFrame(metrics)

    for _, row in metrics_df.iterrows():
        print(
            f'{row["model"]}, top_k={row["top_k"]}: '
            f'EM: {row["EM"]:.2%}, F1: {row["F1"]:.2%}'
        )

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        combined_df.to_excel(writer, sheet_name="Answers", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    main()