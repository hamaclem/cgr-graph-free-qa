import os
import re
import json
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from load_texts import TUPLES_PATH

CACHE_PATH = "data/processed/condition_evals.json"
PROMPT_PATH = "src/prompts/conditionEvaluation_prompt.txt"

BASE_URL = "https://llm3-compute.cms.hu-berlin.de/v1"
MODEL = "llm3"
OPENAI_API_KEY = "secret_but_not_used"

CONCURRENCY = 10
BATCH_SIZE = 30

def gate_status(conditions, evals):
    values = [evals.get(condition) for condition in conditions]
    if any(value is False for value in values):
        return "false"
    if all(value is True for value in values):
        return "true"
    return "unknown"

def load_cache():
    try:
        with open(CACHE_PATH) as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def filter_tuples(group, tuples, evals):
    status = group["tuple_index"].map(lambda i: gate_status(tuples[i].get("conditions") or [], evals))
    return group[status != "false"]

def parse_reply(content, batch):
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())      # remove markdown
    reply = json.loads(content)
    return {condition: reply[condition] for condition in batch if condition in reply and reply[condition] in (True, False, None)}

async def evaluate(query, batch, system_prompt, client, semaphore):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\nConditions: {json.dumps(batch, ensure_ascii=False)}"},
                ],
                temperature=0.0
            )
            return parse_reply(response.choices[0].message.content, batch)
        except Exception as e:
            print(f"Failed ({query[:40]}...): {e!r}")
            return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",
        choices=["medcpt", "hu", "oracle"],
        default="medcpt"
    )
    parser.add_argument(
        "--normalized",
        action=argparse.BooleanOptionalAction,
        default=None
    )
    parser.add_argument(
        "--tuples_k",
        type=int,
        default=50
    )
    args = parser.parse_args()
    if args.normalized is None:
        args.normalized = args.embed == "hu"
    return args.embed, args.normalized, args.tuples_k

async def main():
    embed, normalized, tuples_k = parse_args()
    normalize_suffix = "" if normalized == (embed == "hu") else ("_normalized" if normalized else "_unnormalized")
    retrieval_df = pd.read_excel(f"results/retrieval/retrieval_results_{embed}{normalize_suffix}_tuples.xlsx")
    with open(TUPLES_PATH) as file:
        tuples = json.load(file)
    with open(PROMPT_PATH) as file:
        system_prompt = file.read()
    cache = load_cache()

    jobs = []
    for query, group in retrieval_df.groupby("query", sort=False):
        top = group.sort_values("rank").head(tuples_k)
        conditions = list(dict.fromkeys(c for i in top["tuple_index"] for c in tuples[i].get("conditions") or []))
        todo = [c for c in conditions if c not in cache.get(query, {})]
        jobs += [(query, todo[i:i + BATCH_SIZE]) for i in range(0, len(todo), BATCH_SIZE)]
    print(f"{retrieval_df['query'].nunique()} questions, {sum(len(b) for _, b in jobs)} conditions to evaluate in {len(jobs)} calls")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with AsyncOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY) as client:
        results = await tqdm_asyncio.gather(
            *[evaluate(query, batch, system_prompt, client, semaphore) for query, batch in jobs],
            total=len(jobs), desc="Evaluating conditions"
        )

    asked = answered = failed = 0
    for (query, batch), result in zip(jobs, results):
        asked += len(batch)
        if result is None:
            failed += 1
            continue
        answered += len(result)
        cache.setdefault(query, {}).update(result)

    print(f"{failed} calls failed")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as file:
        json.dump(cache, file, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    asyncio.run(main())