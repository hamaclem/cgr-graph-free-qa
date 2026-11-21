import argparse
import os
import re
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from load_texts import TUPLES_PATH
from gate_tuples import load_cache, filter_tuples

SYSTEM_PROMPT_PATH = "src/prompts/answerGeneration_prompt.txt"
TUPLES_SYSTEM_PROMPT_PATH = "src/prompts/answerGeneration_tuples_prompt.txt"

LLM_CONFIGS = {
    "hu": {
        "base_url": "https://llm3-compute.cms.hu-berlin.de/v1",
        "model": "llm3"
    },
    # Ollama's OpenAI endpoint ignores num_ctx and silently truncates long prompts;
    # set num_ctx 16384 via ollama/*.Modelfile (`ollama create <name> -f <file>`)
    "qwen2.5": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b-16k"
    },
    "llama3.1": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b-16k"
    }
}

OPENAI_API_KEY = "secret_but_not_used"

def create_prompt(query, retrieved_chunks, retrieved_tuples=None):
    context_parts = []

    for _, chunk in (retrieved_chunks.iterrows() if retrieved_chunks is not None else []):
        context_parts.append(
            f"Article: {chunk['article']}\n"
            f"Section: {chunk['section']}\n"
            f"Text: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    if retrieved_tuples is not None and not retrieved_tuples.empty:
        facts = "\n".join(
            f"- {t['text']} [Article: {t['article']}]"
            for _, t in retrieved_tuples.iterrows()
        )
        context = f"{context}\n\nStructured facts:\n{facts}".strip()

    prompt = f"""
        Answer the following biomedical question using the provided evidence.

        Question:
        {query}

        Evidence:
        {context}
        """
    return prompt

def generate_answer(query, retrieved_chunks, system_prompt, client, model, retrieved_tuples=None):
    prompt = create_prompt(query, retrieved_chunks, retrieved_tuples)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False,
    )

    prompt_tokens = response.usage.prompt_tokens if response.usage else None  # shows whether the prompt was truncated
    return response.choices[0].message.content, prompt_tokens

ANSWER_MARKER = re.compile(r"\**\s*answer\s*\**\s*:\s*\**", re.IGNORECASE)  # ANSWER:, **ANSWER:**, **Answer**:

def extract_short_answer(response):
    # last marker wins
    if not isinstance(response, str) or not response:
        return ""

    matches = list(ANSWER_MARKER.finditer(response))
    if not matches:
        return ""

    tail = response[matches[-1].end():].strip()
    first_line = tail.splitlines()[0] if tail else ""
    return first_line.strip().strip("*\"'` ").rstrip(".").strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        choices=LLM_CONFIGS.keys(),
        type=str,
        default="hu"
    )
    parser.add_argument(
        "--embed",
        choices=["medcpt", "hu", "oracle"],  # oracle = gold-article chunks
        type=str,
        default="medcpt"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10
    )
    parser.add_argument(
        "--normalized",
        action=argparse.BooleanOptionalAction,
        default=None  # None = per-model default: hu normalized (cosine), medcpt not (dot product)
    )
    parser.add_argument(
        "--tuples_k",
        type=int,
        default=0
    )
    parser.add_argument(
        "--context",
        choices=["docs", "tuples", "both"],
        type=str,
        default=None  # None = "both" if --tuples_k > 0 else "docs"
    )
    parser.add_argument(
        "--gate",
        action="store_true"
    )

    args = parser.parse_args()

    if args.context is None:
        args.context = "both" if args.tuples_k else "docs"
    if args.context != "docs" and not args.tuples_k:
        parser.error(f"--context {args.context} requires --tuples_k > 0")
    if args.gate and not args.tuples_k:
        parser.error("--gate requires --tuples_k > 0")

    if args.normalized is None:
        args.normalized = args.embed == "hu"

    config = LLM_CONFIGS[args.llm]

    client = OpenAI(
        base_url=config["base_url"],
        api_key=OPENAI_API_KEY,
    )

    return client, config, args.llm, args.embed, args.top_k, args.normalized, args.tuples_k, args.context, args.gate

def main():
    client, config, llm, embed, top_k, normalized, tuples_k, context, gate = parse_args()

    normalize_suffix = "" if normalized == (embed == "hu") else ("_normalized" if normalized else "_unnormalized")  # suffix only when deviating from per-model default
    retrieval_path = f"results/retrieval/retrieval_results_{embed}{normalize_suffix}.xlsx"
    tuples_suffix = f"_tuples{tuples_k}" if tuples_k else ""
    gate_suffix = "_gated" if gate else ""
    if context == "tuples":
        output_path = f"results/answers/{llm}/answers_{llm}_{embed}{normalize_suffix}_tuplesonly{tuples_k}{gate_suffix}.xlsx"
    else:
        output_path = f"results/answers/{llm}/answers_{llm}_{embed}{normalize_suffix}_top{top_k}{tuples_suffix}{gate_suffix}.xlsx"

    retrieval_df = pd.read_excel(retrieval_path)

    tuple_groups = {}
    if tuples_k:
        tuples_df = pd.read_excel(f"results/retrieval/retrieval_results_{embed}{normalize_suffix}_tuples.xlsx")
        tuple_groups = {q: g.sort_values("rank").head(tuples_k) for q, g in tuples_df.groupby("query")}

        # gating
        if gate:
            with open(TUPLES_PATH) as file:
                tuples = json.load(file)
            cache = load_cache()
            tuple_groups = {q: filter_tuples(g, tuples, cache[q]) for q, g in tuple_groups.items()}

    with open(SYSTEM_PROMPT_PATH, "r") as file:
        system_prompt = file.read()
    if tuples_k:
        with open(TUPLES_SYSTEM_PROMPT_PATH, "r") as file:
            tuples_prompt = file.read()
        if context == "tuples":  # no passages in context: facts are the whole evidence, drop rules about passages
            lead = "The evidence may end with structured facts"
            assert lead in tuples_prompt and "text passages" in tuples_prompt, \
                "tuples prompt wording changed: update the tuples-only adjustments in generate_answers.py"
            tuples_prompt = tuples_prompt.replace(lead, "The evidence consists only of structured facts")
            tuples_prompt = "\n".join(l for l in tuples_prompt.splitlines() if "text passages" not in l)
        system_prompt += tuples_prompt

    results = []

    query_groups = retrieval_df.groupby("query", sort=False)
    for query_id, (query, group) in enumerate(
        tqdm(
            query_groups,
            total=retrieval_df["query"].nunique(),
            desc="Generating answers",
            unit="question"
        ), start=1):

        retrieved_chunks = None if context == "tuples" else group.sort_values("rank").head(top_k)
        retrieved_tuples = tuple_groups.get(query, tuples_df.head(0)) if tuples_k else None

        full_answer, prompt_tokens = generate_answer(query, retrieved_chunks, system_prompt, client, config["model"], retrieved_tuples)
        short_answer = extract_short_answer(full_answer)

        results.append({
            "id": query_id,
            "q": query,
            "a": short_answer,
            "r": full_answer,
            "format_ok": bool(short_answer),
            "prompt_tokens": prompt_tokens,
            "n_tuples": len(retrieved_tuples) if retrieved_tuples is not None else 0
        })

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    main()