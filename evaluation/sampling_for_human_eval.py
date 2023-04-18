import pandas as pd
from datasets import load_dataset
csv_files = [
    "csvs/navjordj_t5-base-snl_per_sample_scores.csv",
    "csvs/navjordj_t5-large-snl-2_per_sample_scores.csv",
    "csvs/navjordj_t5-base-cnndaily-2_per_sample_scores.csv",
    "csvs/navjordj_t5-large-cnndaily_per_sample_scores.csv"]

sample_per_model = 5
seed = 42


def print_to_file(print_str, file):
    print(print_str, file=file)


def print_article_summary(name, articles, summaries, labels):
    with open(f"{name}_human_eval.txt", "w", encoding="utf-8") as f:
        print_to_file(f"Model: {name}", f)
        for article, summary, label in zip(articles, summaries, labels):
            print_to_file("====== \n Article", f)
            print_to_file(article, f)
            print_to_file("====== \n label", f)
            print_to_file(label, f)
            print_to_file("====== \n Summary", f)
            print_to_file(summary, f)
            print_to_file("===========================", f)


large_snl = pd.read_csv(csv_files[1]).reset_index()
large_snl = large_snl.sort_values(by="rouge1", ascending=False)
large_snl_top_indices = large_snl.head(sample_per_model).index
large_snl_bottom_indices = large_snl.tail(sample_per_model).index
large_snl_random_indices = large_snl.sample(
    sample_per_model, replace=False, random_state=seed).index


large_cnn = pd.read_csv(csv_files[3]).reset_index()
large_cnn = large_cnn.sort_values(by="rouge1", ascending=False)
print(large_cnn.head(10))
large_cnn_top_indices = large_cnn.head(sample_per_model).index
large_cnn_bottom_indices = large_cnn.tail(sample_per_model).index
large_cnn_random_indices = large_cnn.sample(
    sample_per_model, replace=False, random_state=seed).index


def process_csv(csv_file, article, labels):
    df = pd.read_csv(csv_file)
    df = df.join(article)
    df = df.join(labels)
    # print(csv_file)
    # df["HMean"] = df[["rouge1", "rouge2", "rougeL"]].mean(axis=1)
    df = df.reset_index()

    if "snl" in csv_file:
        random_indices = large_snl_random_indices
        top_indices = large_snl_top_indices
        bottom_indices = large_snl_bottom_indices
    elif "cnndaily" in csv_file:
        random_indices = large_cnn_random_indices
        top_indices = large_cnn_top_indices
        bottom_indices = large_cnn_bottom_indices

    random_n = df[df.index.isin(random_indices)]
    top_n = df[df.index.isin(top_indices)]
    bottom_n = df[df.index.isin(bottom_indices)]

    print(csv_file)

    print_article_summary(
        f"Random_{csv_file.split('/')[1].rstrip('.csv')}",
        random_n["article"].values,
        random_n["summary"].values,
        random_n["labels"].values
    )
    print_article_summary(
        f"Top_{csv_file.split('/')[1].rstrip('.csv')}",
        top_n["article"].values,
        top_n["summary"].values,
        top_n["labels"].values
    )

    print_article_summary(
        f"Bottom_{csv_file.split('/')[1].rstrip('.csv')}",
        bottom_n["article"].values,
        bottom_n["summary"].values,
        bottom_n["labels"].values
    )


if __name__ == "__main__":
    for csv_file in csv_files:
        if "cnndaily" in csv_file:
            df = load_dataset(
                "jkorsvik/cnn_daily_mail_nor_final",
                split="test"
            ).to_pandas().rename(columns={"highlights": "labels"})
        elif "snl" in csv_file:
            df = load_dataset(
                "navjordj/SNL_summarization",
                split="test"
            ).to_pandas().rename(columns={"ingress": "labels"})
        labels = df[["labels"]]
        article = df[["article"]]
        process_csv(csv_file, article, labels)

# Path: evaluation\human_eval.py
