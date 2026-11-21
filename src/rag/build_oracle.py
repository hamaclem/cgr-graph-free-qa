import os
import json
import urllib.parse
import pandas as pd
from load_texts import tuple_to_text, CHUNKS_PATH, TUPLES_PATH

QUERIES_PATH = "data/raw/condmedqa.xlsx"
DOCS_PATH = "data/processed/docs_merged.json"
OUT_CHUNKS = "results/retrieval/retrieval_results_oracle.xlsx"
OUT_TUPLES = "results/retrieval/retrieval_results_oracle_tuples.xlsx"

REDIRECTS = {"aids": "management of hiv/aids", "perfusion scan": "ventilation/perfusion scan"}

def title(x):
    title = urllib.parse.unquote(str(x).rsplit("/wiki/", 1)[-1]).replace("_", " ").casefold()
    return REDIRECTS.get(title, title)

def main():
    queries = pd.read_excel(QUERIES_PATH)
    chunks = json.load(open(CHUNKS_PATH))
    tuples = json.load(open(TUPLES_PATH))
    doc_article = {doc["id"]: doc["metadata"]["article"] for doc in json.load(open(DOCS_PATH))}

    chunk_rows, tuple_rows = [], []
    for q in queries.itertuples():
        golds = [title(q.wiki1), title(q.wiki2)]
        rank = 0
        for gold in golds:
            for i, chunk in enumerate(chunks):
                if title(chunk["metadata"]["article"]) == gold:
                    rank += 1
                    chunk_rows.append({"query": q.q, 
                                       "rank": rank, 
                                       "chunk_index": i,
                                       "article": chunk["metadata"]["article"], 
                                       "section": chunk["metadata"]["section"],
                                       "section_path": " > ".join(chunk["metadata"]["section_path"]), 
                                       "text": chunk["text"]})

        rank = 0
        for i, tuple in enumerate(tuples):
            article = doc_article.get(tuple["source"], "")
            if title(article) in golds:
                rank += 1
                tuple_rows.append({"query": q.q, 
                                   "rank": rank, 
                                   "tuple_index": i,
                                   "doc_id": tuple["source"], "article": article, "text": tuple_to_text(tuple)})

    chunk_df, tuple_df = pd.DataFrame(chunk_rows), pd.DataFrame(tuple_rows)
    os.makedirs(os.path.dirname(OUT_CHUNKS), exist_ok=True)
    chunk_df.to_excel(OUT_CHUNKS, index=False)
    os.makedirs(os.path.dirname(OUT_TUPLES), exist_ok=True)
    tuple_df.to_excel(OUT_TUPLES, index=False)

if __name__ == "__main__":
    main()