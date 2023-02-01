from mediawikiapi import MediaWikiAPI
import pandas as pd
import tqdm
import datetime as dt
import asyncio
import multiprocessing
import os
import psutil
import threading
import time


import os

mediawikiapi = MediaWikiAPI()


mediawikiapi.config.language = "no"
nowiki_df = pd.read_csv(r"datasets\no_wiki\articles.csv", index_col=0)

nowiki_articles = nowiki_df["redirect"].values[:100]

urls = []
headlines = []
articles = []
ingresses = []
categories = []

# Split the list of URLs into batches

batch_size = 1000  # os.cpu_count() - 2
num_cores = os.cpu_count() - 2


async def collect_article_data(name):
    try:
        page = mediawikiapi.page(name)
        title = page.title
        url = page.url
        content = page.content
        summary = page.summary
        categories = page.categories
        # Some if-test to check if the page is a redirected page
        # and if there is a valid hit
        return title, url, content, summary, categories
    except mediawikiapi.exceptions.PageError as e:
        print("Got page error, skipping")
        print(f"Error fetching the article: {name}")
        return None, None, None, None, None


async def await_async_collection(values):
    await asyncio.gather(
        *(
            collect_article_data(value)
            for value in values
        )
    )


def run_async_collection(values):
    return asyncio.run(await_async_collection(values))


def multiprocessing_executor(list_of_names):
    # Not used atm
    start = time.time()
    with multiprocessing.Pool(processes=4) as multiprocessing_pool:
        batch = multiprocessing_pool.map(
            run_async_collection,
            list_of_names,
        )
    end = time.time()
    print(end - start)
    return batch


# multiprocessing_executor()


def main():
    # Split the list of URLs into batches
    wiki_name_batches = [nowiki_articles[i:i + batch_size]
                         for i in range(0, len(nowiki_articles), batch_size)]
    print(wiki_name_batches)

    for url_batch in tqdm.tqdm(wiki_name_batches):

        print(multiprocessing_executor(list(url_batch)))

        """urls.append(url)
        articles.append(content)
        ingresses.append(summary)
        categories.append(category)
        headlines.append(title)
"""


if __name__ == "__main__":
    main()
