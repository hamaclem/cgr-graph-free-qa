from pathlib import Path
import pandas as pd

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
OUTPUT_PATH = "results/evaluation/answers/eval_rag_combined.xlsx"

def main():
    models = ["hu", "qwen2.5", "llama3.1"]
    files = [
        Path(f"results/evaluation/answers/eval_rag_{model}.xlsx")
        for model in models
    ]

    ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)
    combined_df = ground_truth_df[["id", "q", "a"]].rename(
        columns={"a": "ground_truth"}
    )

    metrics = []

    for model, file in zip(models, files):
        df = pd.read_excel(file)

        answers = df[["id", "a"]].rename(columns={"a": f"a_{model}"})
        combined_df = combined_df.merge(answers, on="id")

        metrics.append({
            "model": model,
            "EM": df["em"].mean(),
            "F1": df["f1"].mean()
        })

    metrics_df = pd.DataFrame(metrics)

    for _, row in metrics_df.iterrows():
        print(f'{row["model"]}: EM: {row["EM"]:.2%}, F1: {row["F1"]:.2%}')

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        combined_df.to_excel(writer, sheet_name="Answers", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    main()