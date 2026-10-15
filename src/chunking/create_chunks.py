import json
from transformers import AutoTokenizer

DOCS_PATH = "data/processed/docs_simple.json"
CHUNKS_PATH = "data/processed/chunks.json"

MAX_CHUNK_LENGTH = 512
STRIDE = 64

def get_token_length(tokenizer, text):
    return len(
        tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
    )

def get_max_text_length(tokenizer, context):
    context_tokens = get_token_length(tokenizer, context)

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    return MAX_CHUNK_LENGTH - context_tokens - special_tokens

def create_chunks(tokenizer, doc, start_chunk_id):
    text = doc["text"]

    article = doc["metadata"]["article"].lower()
    section_path = " > ".join(doc["metadata"]["section_path"]).lower()
    context = f"{article}\n{section_path}"

    max_text_length = get_max_text_length(tokenizer, context)

    chunk_tokens_list = tokenizer(
        text,
        max_length=max_text_length,
        truncation=True,
        stride=STRIDE,
        return_overflowing_tokens=True,
        add_special_tokens=False
    )["input_ids"]

    chunks = []

    for i, chunk_tokens in enumerate(chunk_tokens_list):
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        ).strip()

        chunk = {
            "text": chunk_text,
            "metadata": {
                **doc["metadata"],
                "chunk_id": start_chunk_id + i
            }
        }

        chunks.append(chunk)

    return chunks

def create_all_chunks(tokenizer, docs):
    chunks = []
    chunk_id = 0

    for doc in docs:
        doc_chunks = create_chunks(tokenizer, doc, chunk_id)
        chunks.extend(doc_chunks)
        chunk_id += len(doc_chunks)

    return chunks

def main():
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)

    chunks = create_all_chunks(tokenizer, docs)

    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()