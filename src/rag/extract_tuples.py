import json
import asyncio
from openai import AsyncOpenAI

DOCS_PATH = "data/processed/docs_merged.json"
PROMPT_PATH = "src/prompts/tupleExtraction_prompt.txt"
OUTPUT_PATH = "data/processed/tuples.json"

BASE_URL = "https://llm3-compute.cms.hu-berlin.de/v1"
MODEL = "llm3"
OPENAI_API_KEY = "secret_but_not_used"

CONCURRENCY = 5

def create_prompt(doc):
    user_prompt = (
        f"Article: {doc['metadata']['article']}\n"
        f"Section: {' > '.join(doc['metadata']['section_path'])}\n"
        f"Text:\n{doc['text']}"
    )
    
    return user_prompt

async def extract_tuples(doc, system_prompt, client, semaphore):
    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": create_prompt(doc)},
            ],
            temperature=0.0
        )

    return response.choices[0].message.content

async def main():
    with open(DOCS_PATH) as file:
        docs = json.load(file)
    with open(PROMPT_PATH) as file:
        system_prompt = file.read()

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with AsyncOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY) as client:
        tasks = [
            extract_tuples(doc, system_prompt, client, semaphore)
            for doc in docs[:5]
        ]
        results = await asyncio.gather(*tasks)

    tuples = []

    for result in results:
        extracted_tuples = json.loads(result)
        tuples.extend(extracted_tuples)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(tuples, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())