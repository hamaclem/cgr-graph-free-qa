import subprocess
import sys

LLMS = ["hu", "qwen2.5", "llama3.1"]
TOP_K = [1, 3, 5, 10]

for llm in LLMS:
    for k in TOP_K:
        subprocess.run(
            [sys.executable, "generate_answers.py", "--llm", llm, "--top_k", k]
        )