import json
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

QUERIES_PATH = "data/raw/condmedqa.xlsx"
CHUNKS_PATH = "data/processed/chunks_merged.json"
INDEX_PATH = "data/processed/medcpt_faiss.index"
OUTPUT_PATH = "results/retrieval/retrieval_results.xlsx"

TOP_K = 10
BATCH_SIZE = 34

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def encode_queries(queries, tokenizer, model, device):
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

    return embeds.cpu().numpy().astype("float32")

def process_batch(batch_queries, tokenizer, model, device, index, chunks):
    embeds = encode_queries(batch_queries, tokenizer, model, device)

    scores, indices = index.search(embeds, TOP_K)

    results = []

    for query, query_scores, query_indices in zip(batch_queries, scores, indices):
        for rank, (score, chunk_index) in enumerate(
            zip(query_scores, query_indices),
            start=1
        ):
            chunk = chunks[chunk_index]

            results.append({
                "query": query,
                "rank": rank,
                "score": float(score),
                "chunk_index": int(chunk_index),
                "article": chunk["metadata"]["article"],
                "section": chunk["metadata"]["section"],
                "section_path": " > ".join(chunk["metadata"]["section_path"]),
                "text": chunk["text"]
            })
    return results

def main():
    device = get_device()

    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    
    model.eval()
    model.to(device)

    queries = pd.read_excel(QUERIES_PATH)
    queries = queries["q"].tolist()

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    index = faiss.read_index(INDEX_PATH)

    results = []

    for start in range(0, len(queries), BATCH_SIZE):
        batch_queries = queries[start:start + BATCH_SIZE]
        batch_results = process_batch(batch_queries, tokenizer, model, device, index, chunks)
        results.extend(batch_results)

    results_df = pd.DataFrame(results)
    results_df.to_excel(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
