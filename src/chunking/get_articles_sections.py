import asyncio
import wikipediaapi
import json
from urllib.parse import unquote

ARTICLES_LIST_PATH = "data/raw/article_list.txt"
SECTIONS_PATH = "data/raw/articles_sections.json"

# Asynchronous client
wiki_wiki = wikipediaapi.AsyncWikipedia(
    user_agent='Bachelor Thesis: A Graph-Free Approach to Biomedical QA (hanna.clemen@student.hu-berlin.de)', 
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

semaphore = asyncio.Semaphore(10)

EXCLUDED_SECTIONS = {
    "Further reading",
    "See also",
    "References",
    "External links",
    "Notes",
    "Notes and references",
    "Explanatory notes",
    "Bibliography",
    "Sources"
}

async def load_article(raw_title):
    async with semaphore:
        title = unquote(raw_title)
        page = wiki_wiki.page(title)

        if await page.exists():
            return title, {
                "summary": await page.summary,
                "sections": await extract_sections(page)
            }

        return title, None

async def extract_sections(page):
    sections = await page.sections

    def parse_sections(sections):
        result = []

        for section in sections:
            if section.title in EXCLUDED_SECTIONS:
                continue

            subsections = parse_sections(section.sections)
    
            entry = {
                "title": section.title
            }

            if section.text:
                entry["text"] = section.text

            if subsections:
                entry["subsections"] = subsections

            if section.text or subsections:
                result.append(entry)

        return result

    return parse_sections(sections)

async def main():
    with open(ARTICLES_LIST_PATH, "r") as f:
        titles = [line.strip() for line in f]

    tasks = [load_article(title) for title in titles]
    results = await asyncio.gather(*tasks)

    articles = {}
    missing_articles = []

    for title, article in results:
        if article is not None:
            articles[title] = article
        else:
            missing_articles.append(title)

    print("Loaded articles:", len(articles))
    print("Missing articles:", len(missing_articles))

    if missing_articles:
        print(missing_articles)

    with open(SECTIONS_PATH, "w") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())