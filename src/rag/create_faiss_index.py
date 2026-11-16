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
    args = parser.parse_args()
    if args.normalize is None:
        args.normalize = args.embed == "hu"

    return args.embed, args.normalize

def main():
    embed, normalize = parse_model()

    embeddings_path = f"data/processed/{embed}_embeds.npy"
    
    suffix = "" if normalize == (embed == "hu") else ("_normalized" if normalize else "_unnormalized")  # suffix only when deviating from per-model default
    index_path = f"data/processed/{embed}{suffix}_faiss.index" 

    embeds = np.load(embeddings_path).astype(np.float32)

    if normalize:
        faiss.normalize_L2(embeds)
    
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)

    faiss.write_index(index, index_path)

if __name__ == "__main__":
    main()