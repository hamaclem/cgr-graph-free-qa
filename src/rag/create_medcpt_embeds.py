import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

CHUNKS_PATH = "data/processed/chunks_merged.json"
EMBEDDINGS_PATH = "data/processed/medcpt_embeds.npy"

BATCH_SIZE = 32
MAX_CHUNK_LENGTH = 512

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def generate_embeddings(chunks, tokenizer, model, device):
    embeds = []
    
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        
        texts = [
            f"{chunk['metadata']['article'].lower()}\n"
            f"{' > '.join(chunk['metadata']['section_path']).lower()}\n"
            f"{chunk['text']}"
            for chunk in batch
        ]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_CHUNK_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeds = outputs.last_hidden_state[:, 0, :]
        embeds.append(batch_embeds.cpu())

    return torch.cat(embeds).numpy()

def main():
    device = get_device()

    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    model.eval()
    model.to(device)

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    embeds = generate_embeddings(chunks, tokenizer, model, device)

    np.save(EMBEDDINGS_PATH, embeds)

if __name__ == "__main__":
    main()
