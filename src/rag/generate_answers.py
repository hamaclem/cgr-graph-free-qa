import re
import pandas as pd
from openai import OpenAI

RETRIEVAL_PATH = "results/retrieval/retrieval_results.xlsx"
SYSTEM_PROMPT_PATH = "src/prompts/answerGeneration_prompt.txt"
OUTPUT_PATH = "results/answers/rag_answers.xlsx"

BASE_URL = "https://llm3-compute.cms.hu-berlin.de/v1"
MODEL = "llm3"
OPENAI_API_KEY = "secret_but_not_used"

TOP_K = 5

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

def generate_answer(query, retrieved_chunks, system_prompt):
    client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

    prompt = create_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False
    )

    return response.choices[0].message.content

def extract_short_answer(response):
    if not response:
        return ""

    if "ANSWER:" in response:
        return response.split("ANSWER:", 1)[1].strip()

    return response.strip()

def main():
    retrieval_df = pd.read_excel(RETRIEVAL_PATH)

    with open(SYSTEM_PROMPT_PATH, "r") as file:
        system_prompt = file.read()
    
    results = []

    for query_id, (query, group) in enumerate(retrieval_df.groupby("query", sort=False), start=1):
        retrieved_chunks = group.head(TOP_K)

        full_answer = generate_answer(query, retrieved_chunks, system_prompt)
        short_answer = extract_short_answer(full_answer)

        results.append({
            "id": query_id,
            "q": query,
            "a": short_answer,
            "r": full_answer
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()