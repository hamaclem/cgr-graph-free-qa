import subprocess
import sys

LLMS = ["hu", "qwen2.5", "llama3.1"]
EMBEDS = ["medcpt", "hu"]
TOP_K = [1, 3, 5, 10]

for llm in LLMS:
    for embed in EMBEDS:
        for k in TOP_K:
            print(f"Starting model {llm}, with embedding {embed}, top K = {k}")
            subprocess.run(
                [sys.executable, "src/rag/generate_answers.py", "--llm", llm, "--embed", embed, "--top_k", str(k)]
            )