import os
import json
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

DOCS_PATH = "data/processed/docs_merged.json"
PROMPT_PATH = "src/prompts/tupleExtraction_prompt.txt"
OUTPUT_PATH = "data/processed/tuples.json"

BASE_URL = "https://llm3-compute.cms.hu-berlin.de/v1"
MODEL = "llm3"
OPENAI_API_KEY = "secret_but_not_used"

CONCURRENCY = 10

def create_prompt(doc):
    user_prompt = (
        f"Article: {doc['metadata']['article']}\n"
        f"Section: {' > '.join(doc['metadata']['section_path'])}\n"
        f"Text:\n{doc['text']}"
    )
    
    return user_prompt

async def extract_tuples(doc, system_prompt, client, semaphore):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": create_prompt(doc)},
                ],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Failed {doc['id']}: {e!r}")
            return None

async def main():
    with open(DOCS_PATH) as file:
        docs = json.load(file)
    with open(PROMPT_PATH) as file:
        system_prompt = file.read()

    # resume: keep existing tuples, only process docs not yet in the output
    try:
        with open(OUTPUT_PATH) as file:
            tuples = json.load(file)
    except FileNotFoundError:
        tuples = []
    done = {tuple["source"] for tuple in tuples}
    docs = [doc for doc in docs if doc["id"] not in done]
    print(f"{len(done)} docs already done, {len(docs)} to process")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with AsyncOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY) as client:
        results = await tqdm_asyncio.gather(
            *[
                extract_tuples(doc, system_prompt, client, semaphore)
                for doc in docs
            ],
            total=len(docs),
            desc="Extracting tuples"
        )

    failed = [doc["id"] for doc, result in zip(docs, results) if result is None]
    print(f"{len(failed)} failed: {failed}")

    for doc, extracted_tuples in zip(docs, results):
        if extracted_tuples is None:
            continue

        for item in extracted_tuples:
            item["source"] = doc["id"]

        tuples.extend(extracted_tuples)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as file:
        json.dump(tuples, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())