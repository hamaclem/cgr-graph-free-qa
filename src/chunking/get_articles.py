import asyncio
import wikipediaapi
import json
from urllib.parse import unquote

ARTICLES_LIST_PATH = "data/raw/article_list.txt"
ARTICLES_PATH = "data/raw/articles.json"

# Asynchronous client
wiki_wiki = wikipediaapi.AsyncWikipedia(
    user_agent='Bachelor Thesis: A Graph-Free Approach to Biomedical QA (hanna.clemen@student.hu-berlin.de)', 
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

semaphore = asyncio.Semaphore(10)

async def load_article(raw_title):
    async with semaphore:
        title = unquote(raw_title)
        page = wiki_wiki.page(title)

        if await page.exists():
            return title, await page.text

        return title, None

async def main():
    with open(ARTICLES_LIST_PATH, "r") as f:
        titles = [line.strip() for line in f]

    tasks = [load_article(title) for title in titles]
    results = await asyncio.gather(*tasks)

    articles = {}
    missing_articles = []

    for title, text in results:
        if text is not None:
            articles[title] = text
        else:
            missing_articles.append(title)

    print("Loaded articles:", len(articles))
    print("Missing articles:", len(missing_articles))

    if missing_articles:
        print(missing_articles)

    with open(ARTICLES_PATH, "w") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())