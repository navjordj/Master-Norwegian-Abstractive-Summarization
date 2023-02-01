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
import concurrent.futures
from requests import JSONDecodeError
from mediawikiapi.exceptions import PageError

import os

mediawikiapi = MediaWikiAPI()


mediawikiapi.config.language = "no"
nowiki_df = pd.read_csv(r"datasets\no_wiki\articles.csv", index_col=0)

# nowiki_articles = nowiki_df["title"]  # .values[:100]

urls = []
headlines = []
articles = []
ingresses = []
categories = []

# Split the list of URLs into batches

batch_size = 64  # os.cpu_count() - 2
num_cores = os.cpu_count() - 2


def collect_article_data(name):
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
    except JSONDecodeError as e:
        print("Got JSONDecodeError, skipping")

        print(f"Error fetching the article: {name}")
        return None, None, None, None, None
    except KeyError as e:
        print("Got KeyError, skipping")

        print(f"Error fetching the article: {name}")
        return None, None, None, None, None
    except PageError as e:
        print("Got PageErrors, skipping")

        print(f"Error fetching the article: {name}")
        return None, None, None, None, None


async def async_collect_article_data(name):
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
    wiki_name_batches = [nowiki_df.iloc[i:i + batch_size][["title", "redirect"]]
                         for i in range(0, len(nowiki_df), batch_size)]
    # print(wiki_name_batches)

    for url_batch in tqdm.tqdm(wiki_name_batches):

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            res = executor.map(collect_article_data,
                               url_batch["redirect"].values)

            for result in res:
                title, url, content, summary, category = result
                urls.append(url)
                articles.append(content)
                ingresses.append(summary)
                categories.append(category)
                headlines.append(title)

        urls.append(url)
        articles.append(content)
        ingresses.append(summary)
        categories.append(category)
        headlines.append(title)
    date = dt.datetime.now().strftime("%Y-%m-%d")

    dates = [date for _ in range(len(urls))]

    nowiki_collected_df = pd.DataFrame({"url": urls, "date_scraped": dates, "headline": headlines,
                                        "category": categories, "ingress": ingresses, "article": articles})

    print(nowiki_collected_df.shape)
    nowiki_collected_df.to_csv("nowiki_collection.csv", index_label="id")


if __name__ == "__main__":
    main()
