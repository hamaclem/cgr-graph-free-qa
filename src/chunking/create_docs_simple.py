import json

ARTICLES_PATH = "data/raw/articles_sections.json"
DOCS_PATH = "data/processed/docs_simple.json"

def collect_sections(sections, article_title, parent_path=None):
    if parent_path is None:
        parent_path = []

    docs = []

    for section in sections:
        current_path = parent_path + [section["title"]]

        text = section.get("text", "")

        if text:
            metadata = {
                "article": article_title,
                "section": section["title"]
            }

            metadata["section_path"] = current_path

            docs.append({
                "text": text,
                "metadata": metadata
            })

        docs.extend(collect_sections(section.get("subsections", []), article_title, current_path))

    return docs

def main():
    with open(ARTICLES_PATH, "r") as f:
        articles = json.load(f)

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

        docs.extend(collect_sections(article["sections"], title))

    with open(DOCS_PATH, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
   main()