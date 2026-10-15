import numpy as np
import faiss

EMBEDDINGS_PATH = "data/processed/medcpt_embeds.npy"
INDEX_PATH = "data/processed/medcpt_faiss.index"

def main():
    embeds = np.load(EMBEDDINGS_PATH)
    index = faiss.IndexFlatIP(768)
    index.add(embeds)
    faiss.write_index(index, INDEX_PATH)

if __name__ == "__main__":
    main()