import subprocess
import sys

LLMS = ["hu", "qwen2.5", "llama3.1"]

for llm in LLMS:
    subprocess.run(
        [sys.executable, "generate_answers.py", "--llm", llm]
    )