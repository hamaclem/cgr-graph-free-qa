import subprocess
import sys

LLMS = ["hu", "qwen2.5", "llama3.1"]
EMBEDS = ["hu", "medcpt"]
TOP_K = [1, 3, 5, 10]
TUPLES_K = [0, 5, 10]

for llm in LLMS:
    for embed in EMBEDS:
        for k in TOP_K:
            for tuples_k in TUPLES_K:
                print(f"Starting model {llm}, with embedding {embed}, top K = {k}, tuples K = {tuples_k}")

                command = [sys.executable, "src/rag/generate_answers.py", "--llm", llm, "--embed", embed, "--top_k", str(k), "--tuples_k", str(tuples_k)]

                subprocess.run(command)
