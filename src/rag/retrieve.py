import os
import argparse
import json
import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from encode_queries import encode_medcpt_queries, encode_hu_queries
from load_texts import tuple_to_text, CHUNKS_PATH, TUPLES_PATH

QUERIES_PATH = "data/raw/condmedqa.xlsx"
DOCS_PATH = "data/processed/docs_merged.json"

TOP_K = 10
BATCH_SIZE = 34

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def chunk_fields(chunk):
    return {
        "article": chunk["metadata"]["article"],
        "section": chunk["metadata"]["section"],
        "section_path": " > ".join(chunk["metadata"]["section_path"]),
        "text": chunk["text"]
    }

def tuple_fields(tuple):
    return {"doc_id": tuple["source"], "article": tuple["article"], "text": tuple_to_text(tuple)}

def process_batch(batch_queries, encode_queries, index, chunks, normalized, item_fields):
    embeds = encode_queries(batch_queries)

    if normalized:
        faiss.normalize_L2(embeds)

    scores, indices = index.search(embeds, TOP_K)

    results = []

    for query, query_scores, query_indices in zip(batch_queries, scores, indices):
        for rank, (score, chunk_index) in enumerate(
            zip(query_scores, query_indices),
            start=1
        ):
            results.append({
                "query": query,
                "rank": rank,
                "score": float(score),
                "chunk_index": int(chunk_index),
                **item_fields(chunks[chunk_index])
            })
    return results

def parse_embedding_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",
        choices=["medcpt", "hu"],
        default="medcpt"
    )
    parser.add_argument(
        "--normalized",
        action=argparse.BooleanOptionalAction,
        default=None  # None = per-model default: hu normalized (cosine), medcpt not (dot product)
    )
    parser.add_argument(
        "--source", 
        choices=["chunks", "tuples"],
        default="chunks"
    )

    args = parser.parse_args()
    tuple_infix = "_tuple" if args.source == "tuples" else ""

    if args.normalized is None:
        args.normalized = args.embed == "hu"

    if args.embed == "medcpt":
        device = get_device()
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

        model.eval()
        model.to(device)

        def encode(batch_queries):
            return encode_medcpt_queries(batch_queries, tokenizer, model, device)

        infix = "_normalized" if args.normalized else ""
        index_path = f"data/processed/medcpt{tuple_infix}{infix}_faiss.index"
            
    else:
        encode = encode_hu_queries

        infix = "" if args.normalized else "_unnormalized"
        index_path = f"data/processed/hu{tuple_infix}{infix}_faiss.index"

    return args.embed, args.normalized, encode, index_path, args.source

def main():
    embed, normalized, encode, index_path, source = parse_embedding_model()
    queries = pd.read_excel(QUERIES_PATH)
    queries = queries["q"].tolist()

    items_path, item_fields = (TUPLES_PATH, tuple_fields) if source == "tuples" else (CHUNKS_PATH, chunk_fields)
    with open(items_path, "r") as f:
        chunks = json.load(f)

    if source == "tuples":
        with open(DOCS_PATH, "r") as f:
            doc_articles = {doc["id"]: doc["metadata"]["article"] for doc in json.load(f)}
        for tuple in chunks:
            tuple["article"] = doc_articles[tuple["source"]]

    index = faiss.read_index(index_path)
    
    results = []

    for start in range(0, len(queries), BATCH_SIZE):
        batch_queries = queries[start:start + BATCH_SIZE]
        results.extend(process_batch(batch_queries, encode, index, chunks, normalized, item_fields))

    suffix = "" if normalized == (embed == "hu") else ("_normalized" if normalized else "_unnormalized")  # suffix only when deviating from per-model default
    tuple_suffix = "_tuples" if source == "tuples" else ""
    output_path = f"results/retrieval/retrieval_results_{embed}{suffix}{tuple_suffix}.xlsx"

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_excel(output_path, index=False)
    
if __name__ == "__main__":
    main()