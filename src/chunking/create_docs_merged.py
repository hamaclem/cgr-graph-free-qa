import json
from transformers import AutoTokenizer

ARTICLES_PATH = "data/raw/articles_sections.json"
DOCS_PATH = "data/processed/docs_merged.json"

MAX_CHUNK_LENGTH = 512

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

def create_doc(text, article_title, section_path):
    return {
        "text": text,
        "metadata": {
            "article": article_title,
            "section": section_path[-1],
            "section_path": section_path
        }
    }

def save_current_subsections(docs, current_subsections, article_title, parent_path):
    if not current_subsections:
            return
    
    if len(current_subsections) > 1:
        merged_text = "\n\n".join(f"{subsection['title']}\n{subsection['text']}" for subsection in current_subsections)
        docs.append(create_doc(merged_text, article_title, parent_path))

    else:
        subsection = current_subsections[0]
        text = subsection["text"]
        section_path = parent_path + [subsection["title"]]
        docs.append(create_doc(text, article_title, section_path))

def merge_subsections(tokenizer, subsections, article_title, parent_path):
    docs = []

    section_path = " > ".join(parent_path).lower()
    context = f"{article_title.lower()}\n{section_path}"

    max_text_length = get_max_text_length(tokenizer, context)

    current_subsections = []
    current_tokens = 0

    for subsection in subsections:
        title = subsection["title"]
        text = subsection.get("text", "")

        if not text:
            continue

        subsection_text = f"{title}\n{text}"
        subsection_tokens = get_token_length(tokenizer, subsection_text)

        # case 1: subsection is too large to fit in a chunk by itself
        # -> save any collected subsections and keep this one as its own document (handled by create_chunks.py)
        if subsection_tokens > max_text_length:
            save_current_subsections(docs, current_subsections, article_title, parent_path)

            current_subsections = []
            current_tokens = 0

            docs.append(create_doc(text, article_title, parent_path + [title]))

            continue

        separator_tokens = (
            get_token_length(tokenizer, "\n\n")
            if current_subsections
            else 0
        )

        # case 2: adding this subsection would exceed the token limit
        # -> save the collected group, then start a new group with this subsection
        if current_subsections and current_tokens + separator_tokens + subsection_tokens > max_text_length:
            save_current_subsections(docs, current_subsections, article_title, parent_path)

            current_subsections = []
            current_tokens = 0
            separator_tokens = 0

        # case 3: adding this subsection stays within the token limit
        current_subsections.append({
            "title": title,
            "text": text
        })

        current_tokens += separator_tokens + subsection_tokens

    save_current_subsections(docs, current_subsections, article_title, parent_path)

    return docs

def collect_sections(tokenizer, sections, article_title, parent_path=None):
    if parent_path is None:
        parent_path = []

    docs = []

    for section in sections:
        current_path = parent_path + [section["title"]]
        text = section.get("text", "")

        if text:
            docs.append(create_doc(text, article_title, current_path))

        subsections = section.get("subsections", [])

        if subsections:
            docs.extend(merge_subsections(tokenizer, subsections, article_title, current_path))

    return docs

def create_docs(tokenizer, articles):
    docs = []

    for title, article in articles.items():
        docs.append({
            "text": article["summary"],
            "metadata": {
                "article": title,
                "section": "Summary",
                "section_path": ["Summary"]
            }
        })

        docs.extend(collect_sections(tokenizer, article["sections"], title))

    return docs

def main():
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
 
    with open(ARTICLES_PATH, "r") as f:
        articles = json.load(f)

    docs = create_docs(tokenizer, articles)

    with open(DOCS_PATH, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
