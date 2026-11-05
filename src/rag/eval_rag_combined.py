from pathlib import Path
import pandas as pd

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
OUTPUT_PATH = "results/evaluation/answers/eval_rag_combined.xlsx"

def main():
    llms = ["hu"]
    embeds = ["medcpt", "hu"]
    top_ks = [1, 3, 5, 10]
    normalization = [False, True]

    ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)
    combined_df = ground_truth_df[["id", "q", "a"]].rename(
        columns={"a": "ground_truth"}
    )

    metrics = []

    for model in llms:
        for embed in embeds:
            for normalized in normalization:
                suffix = "_normalized" if normalized else ""
                for top_k in top_ks:
                    file = Path(f"results/evaluation/answers/eval_rag_{model}_{embed}{suffix}_top{top_k}.xlsx")

                    df = pd.read_excel(file)

                    answers = df[["id", "a"]].rename(
                        columns={"a": f"a_{model}_{embed}{suffix}_top{top_k}"}
                    )
                    combined_df = combined_df.merge(answers, on="id", how="left")

                    metrics.append({
                        "model": model,
                        "embedding_model": embed,
                        "normalized": normalized,
                        "top_k": top_k,
                        "EM": df["em"].mean(),
                        "F1": df["f1"].mean()
                    })

    metrics_df = pd.DataFrame(metrics)

    for _, row in metrics_df.iterrows():
        print(
            f'{row["model"]}, embedding={row["embedding_model"]}, '
            f'normalized={row["normalized"]}, '
            f'top_k={row["top_k"]}: '
            f'EM: {row["EM"]:.2%}, F1: {row["F1"]:.2%}'
        )

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        combined_df.to_excel(writer, sheet_name="Answers", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    main()