import os
import argparse
import pandas as pd

def load_run(name):
    llm = name.split("_")[0]
    eval_df = pd.read_excel(f"results/evaluation/answers/{llm}/eval_rag_{name}.xlsx", sheet_name="Answers")
    answers_df = pd.read_excel(f"results/answers/{llm}/answers_{name}.xlsx")
    return eval_df.merge(answers_df[["id", "r"]], on="id")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a"
    )
    parser.add_argument(
        "--b"
    )
    args = parser.parse_args()

    a, b = load_run(args.a), load_run(args.b)
    df = a.merge(b[["id", "a", "em", "f1", "r"]], on="id", suffixes=("_A", "_B"))

    categories = {
        f"only_{args.a}_correct": (df["em_A"] == 1) & (df["em_B"] == 0),
        f"only_{args.b}_correct": (df["em_A"] == 0) & (df["em_B"] == 1),
        "both_wrong": (df["em_A"] == 0) & (df["em_B"] == 0),
        "both_correct": (df["em_A"] == 1) & (df["em_B"] == 1),
    }

    summary = pd.DataFrame([{"category": key, "count": int(mask.sum())} for key, mask in categories.items()])
    print(summary.to_string(index=False))

    cols = ["id", "q", "ground_truth", "a_A", "a_B", "f1_A", "f1_B", "r_A", "r_B"]
    names = {"q": "question", "a_A": f"a_{args.a}", "a_B": f"a_{args.b}",
             "f1_A": f"f1_{args.a}", "f1_B": f"f1_{args.b}", "r_A": f"r_{args.a}", "r_B": f"r_{args.b}"}
    output_path = f"results/evaluation/answers/compare_{args.a}_vs_{args.b}.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for name, mask in categories.items():
            if name != "both_correct":
                df.loc[mask, cols].rename(columns=names).to_excel(writer, sheet_name=name, index=False)

if __name__ == "__main__":
    main()