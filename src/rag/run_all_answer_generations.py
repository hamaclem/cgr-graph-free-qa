import subprocess
import sys

LLMS = ["hu"]
EMBEDS = ["hu"]
NORMALIZE = [True]
TOP_K = [1, 3, 5]

for llm in LLMS:
    for embed in EMBEDS:
        for k in TOP_K:
            for normalized in NORMALIZE:
                print(f"Starting model {llm}, with embedding {embed}, top K = {k}, normalized = {normalized}")

                command = [sys.executable, "src/rag/generate_answers.py", "--llm", llm, "--embed", embed, "--top_k", str(k)]

                if normalized:
                    command.append("--normalized")

                subprocess.run(command)