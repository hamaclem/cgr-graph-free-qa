import argparse
import json
import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from encode_queries import encode_medcpt_queries, encode_hu_queries

QUERIES_PATH = "data/raw/condmedqa.xlsx"
CHUNKS_PATH = "data/processed/chunks_merged.json"

TOP_K = 10
BATCH_SIZE = 34

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def process_batch(batch_queries, encode_queries, index, chunks):
    embeds = encode_queries(batch_queries)

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

def parse_embedding_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",
        choices=["medcpt", "hu"],
        default="medcpt"
    )
    args = parser.parse_args()

    if args.embed == "medcpt":
        device = get_device()
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

        model.eval()
        model.to(device)

        def encode(batch_queries):
            return encode_medcpt_queries(batch_queries, tokenizer, model, device)

        index_path = "data/processed/medcpt_faiss.index"

    else:
        encode = encode_hu_queries
        index_path = "data/processed/hu_faiss.index"

    return args.embed, encode, index_path

def main():
    embed, encode, index_path = parse_embedding_model()
    queries = pd.read_excel(QUERIES_PATH)
    queries = queries["q"].tolist()

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    index = faiss.read_index(index_path)
    
    results = []

    for start in range(0, len(queries), BATCH_SIZE):
        batch_queries = queries[start:start + BATCH_SIZE]
        results.extend(process_batch(batch_queries, encode, index, chunks))

    output_path = f"results/retrieval/retrieval_results_{embed}.xlsx"

    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)
    
if __name__ == "__main__":
    main()