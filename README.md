# cgr-graph-free-qa

RAG pipeline for conditional medical QA ([CondMedQA](data/raw/condmedqa.xlsx)): fetch articles, chunk them, extract (entity, relation, entity, conditions) tuples, embed chunks and tuples (MedCPT or HU-hosted model), retrieve each with FAISS, generate answers with an LLM from chunks (optionally plus tuples), and evaluate retrieval and answers.

All scripts are run from the repo root.

## 1. Data & chunking (`src/chunking/`)

| Script | Output |
|---|---|
| `get_articles.py` | `data/raw/articles.json` |
| `get_articles_sections.py` | `data/raw/articles_sections.json` |
| `create_docs_simple.py` / `create_docs_merged.py` | `data/processed/docs_{simple,merged}.json` |
| `create_chunks.py` | `data/processed/chunks_{simple,merged}.json` |
| `token_analysis.py`, `chunking_analysis.py` | stats in `results/evaluation/chunking/` |

Retrieval uses `data/processed/chunks_merged.json`.

## 2. Tuple extraction (`src/tuples/`)

```bash
python src/tuples/extract_tuples.py        # docs_merged.json -> data/processed/tuples.json
python src/tuples/extract_docs_examples.py # sample docs -> data/processed/docs_examples.txt
```

Uses `src/prompts/tupleExtraction_prompt.txt`. Each tuple stores its doc in `source` (= `doc_id`). Re-running resumes: docs already in `tuples.json` are skipped.

## 3. Embeddings & index (`src/rag/`)

```bash
python src/rag/create_medcpt_embeds.py [--source tuples]   # -> data/processed/medcpt[_tuple]_embeds.npy
python src/rag/create_hu_embeds.py [--source tuples]       # -> data/processed/hu[_tuple]_embeds.npy
python src/rag/create_faiss_index.py --embed {medcpt,hu} [--source tuples]   # -> data/processed/<embed>[_tuple]_faiss.index
```

`--source` is `chunks` (default) or `tuples`. Texts to embed come from `src/rag/load_texts.py`; a tuple is embedded as `entity1 relation entity2 (conditions: ...)`.

`--normalize/--no-normalize` (also `--normalized` on later steps): default normalizes `hu` (cosine) but not `medcpt` (dot product). Non-default choices add a suffix to output filenames.

## 4. Retrieval

```bash
python src/rag/retrieve.py --embed {medcpt,hu} [--source tuples]        # -> results/retrieval/retrieval_results_<embed>[_tuples].xlsx
python src/rag/eval_retrieval.py --embed {medcpt,hu} [--source tuples]  # -> results/evaluation/retrieval/eval_retrieval_<embed>[_tuples].xlsx
```

## 5. Answer generation

```bash
python src/rag/generate_answers.py --llm {hu,qwen2.5,llama3.1} --embed {medcpt,hu} --top_k {1,3,5,10} [--tuples_k N]
python src/rag/run_all_answer_generations.py   # loops over LLMS, EMBEDS, TOP_K, TUPLES_K
```

Output: `results/answers/answers_<llm>_<embed>_top<k>[_tuples<N>].xlsx`. System prompt: `src/prompts/answerGeneration_prompt.txt`.

## 6. Answer evaluation

```bash
python src/rag/eval_rag.py --llm {hu,qwen2.5,llama3.1} --embed {medcpt,hu} --top_k {1,3,5,10} [--tuples_k N]   # -> results/evaluation/answers/eval_rag_*.xlsx
python src/rag/eval_rag_combined.py   # -> eval_rag_combined.xlsx (all runs + metrics)
```

`eval_rag_combined.py` loops over `llms`, `embeds`, `top_ks`, `tuples_ks`
