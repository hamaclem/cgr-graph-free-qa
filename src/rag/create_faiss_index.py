import os
import argparse
import numpy as np
import faiss

def parse_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",
        choices=["medcpt", "hu"],
        required=True
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None  # None = per-model default: hu normalized (cosine), medcpt not (dot product)
    )
    parser.add_argument(
        "--source", 
        choices=["chunks", "tuples"], 
        default="chunks"
    )
    args = parser.parse_args()
    if args.normalize is None:
        args.normalize = args.embed == "hu"

    return args.embed, args.normalize, args.source

def main():
    embed, normalize, source = parse_model()

    tuple_infix = "_tuple" if source == "tuples" else ""
    embeddings_path = f"data/processed/{embed}{tuple_infix}_embeds.npy"

    suffix = "" if normalize == (embed == "hu") else ("_normalized" if normalize else "_unnormalized")  # suffix only when deviating from per-model default
    index_path = f"data/processed/{embed}{tuple_infix}{suffix}_faiss.index"

    embeds = np.load(embeddings_path).astype(np.float32)

    if normalize:
        faiss.normalize_L2(embeds)
    
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

if __name__ == "__main__":
    main()