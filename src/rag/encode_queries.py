import numpy as np
import torch
from openai import OpenAI

BASE_URL = "https://llm5-compute.cms.hu-berlin.de/v1"
MODEL = "llm5"
OPENAI_API_KEY = "secret_but_not_used"

def encode_medcpt_queries(queries, tokenizer, model, device):
    with torch.no_grad():
        encoded = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=64
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        embeds = model(**encoded).last_hidden_state[:, 0, :]

    return embeds.cpu().numpy().astype(np.float32)

def encode_hu_queries(queries):
    client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=MODEL,
        input=queries,
    )

    data = sorted(
        response.data,
        key=lambda item: item.index
    )

    return np.asarray(
        [item.embedding for item in data],
        dtype=np.float32
    )