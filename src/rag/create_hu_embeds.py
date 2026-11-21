import os
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from load_texts import load_texts

BASE_URL = "https://llm5-compute.cms.hu-berlin.de/v1"
MODEL = "llm5"
OPENAI_API_KEY = "secret_but_not_used"

BATCH_SIZE = 32

def generate_embeddings(texts):
    client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

    embeds = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), total=(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE, desc="Generating embeddings", unit="batch"):
        batch = texts[start:start + BATCH_SIZE]

        response = client.embeddings.create(
            model=MODEL,
            input=batch
        )

        batch_data = sorted(
            response.data,
            key=lambda item: item.index
        )

        batch_embeddings = np.asarray(
            [item.embedding for item in batch_data],
            dtype=np.float32
        )

        embeds.append(batch_embeddings)

    return np.concatenate(embeds, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", 
        choices=["chunks", "tuples"], 
        default="chunks"
    )
    source = parser.parse_args().source

    embeds = generate_embeddings(load_texts(source))

    infix = "_tuple" if source == "tuples" else ""
    os.makedirs("data/processed", exist_ok=True)
    np.save(f"data/processed/hu{infix}_embeds.npy", embeds)

if __name__ == "__main__":
    main()