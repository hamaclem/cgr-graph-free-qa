import json
import numpy as np
from openai import OpenAI

CHUNKS_PATH = "data/processed/chunks_merged.json"
EMBEDDINGS_PATH = "data/processed/hu_embeds.npy"

BASE_URL = "https://llm5-compute.cms.hu-berlin.de/v1"
MODEL = "llm5"
OPENAI_API_KEY = "secret_but_not_used"

BATCH_SIZE = 32

def generate_embeddings(chunks):
    client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

    embeds = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            f"{chunk['metadata']['article'].lower()}\n"
            f"{' > '.join(chunk['metadata']['section_path']).lower()}\n"
            f"{chunk['text']}"
            for chunk in batch
        ]

        response = client.embeddings.create(
            model=MODEL,
            input=texts
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
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    embeds = generate_embeddings(chunks)

    np.save(EMBEDDINGS_PATH, embeds)

if __name__ == "__main__":
    main()