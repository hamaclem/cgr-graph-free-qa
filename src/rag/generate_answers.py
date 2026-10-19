import argparse
import pandas as pd
from openai import OpenAI

RETRIEVAL_PATH = "results/retrieval/retrieval_results.xlsx"
SYSTEM_PROMPT_PATH = "src/prompts/answerGeneration_prompt.txt"

LLM_CONFIGS = {
    "hu": {
        "base_url": "https://llm3-compute.cms.hu-berlin.de/v1",
        "model": "llm3"
    },
    "qwen2.5": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b"
    },
    "llama3.1": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b"
    }
}

OPENAI_API_KEY = "secret_but_not_used"

def create_prompt(query, retrieved_chunks):
    context_parts = []

    for _, chunk in retrieved_chunks.iterrows():
        context_parts.append(
            f"Article: {chunk['article']}\n"
            f"Section: {chunk['section']}\n"
            f"Text: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
        Answer the following biomedical question using the provided evidence.

        Question:
        {query}

        Evidence:
        {context}
        """
    return prompt

def generate_answer(query, retrieved_chunks, system_prompt, client, model):
    prompt = create_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False,
    )

    return response.choices[0].message.content

def extract_short_answer(response):
    if not response:
        return ""

    if "ANSWER:" in response:
        return response.split("ANSWER:", 1)[1].strip()

    return response.strip()

def parse_llm():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        choices=LLM_CONFIGS.keys(),
        default="hu"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    config = LLM_CONFIGS[args.llm]

    client = OpenAI(
        base_url=config["base_url"],
        api_key=OPENAI_API_KEY,
    )

    return client, config, args.llm, args.top_k
    
def main():
    client, config, llm, top_k = parse_llm()

    output_path = f"results/answers/rag_answers_{llm}_top{top_k}.xlsx"
    
    retrieval_df = pd.read_excel(RETRIEVAL_PATH)

    with open(SYSTEM_PROMPT_PATH, "r") as file:
        system_prompt = file.read()

    results = []

    for query_id, (query, group) in enumerate(retrieval_df.groupby("query", sort=False), start=1):
        retrieved_chunks = group.head(top_k)

        full_answer = generate_answer(query, retrieved_chunks, system_prompt, client, config["model"])
        short_answer = extract_short_answer(full_answer)

        results.append({
            "id": query_id,
            "q": query,
            "a": short_answer,
            "r": full_answer
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    main()