import argparse
import numpy as np
import faiss

def parse_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["medcpt", "hu"],
        required=True
    )
    parser.add_argument(
        "--normalize",
        action="store_true"
    )
    args = parser.parse_args()

    return args.model, args.normalize

def main():
    model, normalize = parse_model()

    embeddings_path = f"data/processed/{model}_embeds.npy"
    index_path = f"data/processed/{model}_faiss.index"

    embeds = np.load(embeddings_path).astype(np.float32)

    if normalize:
        faiss.normalize_L2(embeds)
    
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)

    faiss.write_index(index, index_path)

if __name__ == "__main__":
    main()