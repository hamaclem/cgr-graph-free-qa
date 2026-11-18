import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from load_texts import load_texts

BATCH_SIZE = 32
MAX_CHUNK_LENGTH = 512

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def generate_embeddings(texts, tokenizer, model, device):
    embeds = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", 
        choices=["chunks", "tuples"], 
        default="chunks"
    )
    source = parser.parse_args().source

    device = get_device()

    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    model.eval()
    model.to(device)

    embeds = generate_embeddings(load_texts(source), tokenizer, model, device)

    infix = "_tuple" if source == "tuples" else ""
    np.save(f"data/processed/medcpt{infix}_embeds.npy", embeds)

if __name__ == "__main__":
    main()