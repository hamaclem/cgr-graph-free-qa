import json

CHUNKS_PATH = "data/processed/chunks_merged.json"
TUPLES_PATH = "data/processed/tuples.json"

def tuple_to_text(tuple):
    text = f"{tuple['entity1']} {tuple['relation'].replace('_', ' ')} {tuple['entity2']}"
    if tuple.get("conditions"):
        text += f" (conditions: {'; '.join(tuple['conditions'])})"
    return text

def chunk_to_text(chunk):
    return (
        f"{chunk['metadata']['article'].lower()}\n"
        f"{' > '.join(chunk['metadata']['section_path']).lower()}\n"
        f"{chunk['text']}"
    )

def load_texts(source):
    path = TUPLES_PATH if source == "tuples" else CHUNKS_PATH
    with open(path) as f:
        items = json.load(f)
    to_text = tuple_to_text if source == "tuples" else chunk_to_text
    return [to_text(item) for item in items]