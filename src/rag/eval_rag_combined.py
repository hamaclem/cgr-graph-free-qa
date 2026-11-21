import glob
import os
import re
import pandas as pd

GROUND_TRUTH_PATH = "data/raw/condmedqa.xlsx"
OUTPUT_PATH = "results/evaluation/answers/eval_rag_combined.xlsx"
NAME = re.compile(
    r"(?P<llm>[^_]+)_(?P<embed>[^_]+)(?P<norm>_normalized|_unnormalized)?"
    r"_(?:tuplesonly(?P<only>\d+)|top(?P<top_k>\d+)(?:_tuples(?P<tuples>\d+))?)(?P<gated>_gated)?$"
)

def parse(name):
    match = NAME.match(name)
    if not match:
        return None
    tuples_k = int(match["only"] or match["tuples"] or 0)
    return {
        "run": name,
        "llm": match["llm"],
        "embed": match["embed"] + (match["norm"] or ""),
        "top_k": None if match["only"] else int(match["top_k"]),
        "tuples_k": tuples_k,
        "context": "tuples" if match["only"] else ("both" if tuples_k else "docs"),
        "gate": "gated" if match["gated"] else "none",
    }

def main():
    combined_df = pd.read_excel(GROUND_TRUTH_PATH)[["id", "q", "a"]].rename(columns={"a": "ground_truth"})
    metrics = []

    for path in sorted(glob.glob("results/evaluation/answers/*/eval_rag_*.xlsx")):
        name = os.path.basename(path)[len("eval_rag_"):-len(".xlsx")]
        run = parse(name)

        df = pd.read_excel(path, sheet_name="Answers", keep_default_na=False)
        answers_path = f"results/answers/{run['llm']}/answers_{name}.xlsx"
        answers = (pd.read_excel(answers_path, keep_default_na=False) if os.path.exists(answers_path) else pd.DataFrame())

        metrics.append({
            **run,
            "EM": df["em"].mean(),
            "EM_lenient": df["em_lenient"].mean() if "em_lenient" in df else None,
            "F1": df["f1"].mean(),
            "format_failures": int((~answers["format_ok"].astype(bool)).sum()) if "format_ok" in answers else None,
            "prompt_tokens_max": answers["prompt_tokens"].max() if "prompt_tokens" in answers else None,
            "n_tuples_mean": answers["n_tuples"].mean() if "n_tuples" in answers else None,  # tuples left after gating
            "n_tuples_zero": int((answers["n_tuples"] == 0).sum()) if "n_tuples" in answers else None,
        })
        combined_df = combined_df.merge(df[["id", "a"]].rename(columns={"a": f"a_{name}"}), on="id", how="left")

    metrics_df = pd.DataFrame(metrics).sort_values(["llm", "embed", "context", "top_k", "tuples_k", "gate"], na_position="last")
    print(metrics_df[["run", "EM", "EM_lenient", "F1"]].to_string(index=False, float_format=lambda x: f"{x:.2%}"))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        combined_df.to_excel(writer, sheet_name="Answers", index=False)

if __name__ == "__main__":
    main()