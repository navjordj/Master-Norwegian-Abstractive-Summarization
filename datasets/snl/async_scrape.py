import bs4
import pandas as pd 
import grequests

import re
import tqdm
import datetime as dt


import os

def get_headline(site):
    return site.find("h1", "page-title__heading").text.strip()

def get_article_contents(site):
    elements = site.find_all(["div", "h2"], class_=lambda c: c in ["article-text article-text--snl", "l-article__subheading"])

    article = [element.text.strip().replace("\n", " ") for element in elements]
    return article

def get_categories(site) -> str:
    return ",".join([category.text.strip() for category in site.findAll("li", "breadcrumbs__item")])

    

def main():


    urls_df = pd.read_csv(r"C:\Users\navjo\Documents\summarization_master\datasets\snl\snl_valid_urls.csv", index_col=0)

    snl_urls = urls_df["url"].values

    urls = []
    headlines = []
    articles = []
    ingresses = []
    categories = []

    # Split the list of URLs into batches

    batch_size = 32 

    # Split the list of URLs into batches
    url_batches = [snl_urls[i:i + batch_size] for i in range(0, len(snl_urls), batch_size)]

    for url_batch in tqdm.tqdm(url_batches):

        rs = (grequests.get(u) for u in url_batch)
        responses = grequests.map(rs)

        for r in responses:
            if r and r.status_code == 200:
                site = bs4.BeautifulSoup(r.text)

                content = get_article_contents(site)

                try:
                    ingress, article = content[0], " ".join(content[1:])
                except Exception as e:
                    print(e)
                    print(f"Ingress, article split failed for {r.url}")

                # Filter out short articles
                
                if len(ingress) < 150 or len(article) < 400:
                    continue

                category = get_categories(site)
                headline = get_headline(site)

                urls.append(r.url)
                articles.append(article)
                ingresses.append(ingress)
                categories.append(category)
                headlines.append(headline)




            else:
                print(f"Error fetching URL: {r.url if r else None}")


    date = dt.datetime.now().strftime("%Y-%m-%d")

    dates = [date for _ in range(len(urls))]

    snl_df = pd.DataFrame({"url": urls, "date_scraped": dates, "headline": headlines, "category": categories, "ingress": ingresses, "article": articles})


    print(snl_df.shape)
    snl_df.to_csv("snl.csv", index_label="id")
if __name__ == "__main__":
    main()