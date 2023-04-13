import re
import pandas as pd
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from rouge import Rouge
import evaluate
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

# Function to summarize the text using TextRank


def textrank_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("norwegian"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in sentences])


# Function to summarize the text using Lead-3

def lead_3_summary(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return " ".join(sentences[:3])


metric = evaluate.load("rouge")

# Calculate Rouge scores for the summaries


def rouge_scores(preds, labels, tokenizer=word_tokenize):
    result = metric.compute(
        predictions=preds, references=labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len_words"] = np.mean([len(tokenizer(pred)) for pred in preds])
    return result


def main(dataset_path):
    # Load the dataset
    dataset = load_dataset(dataset_path, split="test")
    # Create a DataFrame
    df = dataset.to_pandas()

    # Apply Lead-3 to the DataFrame
    df['Lead_3_summary'] = df['article'].apply(lead_3_summary)
    # write the results to a txt file where each line is a summary
    with open(f"{dataset_path.split('/')[-1]}_Lead_3_summary.txt", "w", encoding="utf-8") as f:
        for summary in df['Lead_3_summary']:
            f.write(summary + " ")
            f.write("\n")

    # Apply TextRank to the DataFrame
    df['TextRank_summary'] = df['article'].apply(textrank_summary)
    # write the results to a txt file where each line is a summary
    with open(f"{dataset_path.split('/')[-1]}_TextRank_summary.txt", "w", encoding="utf-8") as f:
        for summary in df['TextRank_summary']:
            f.write(summary + " ")
            f.write("\n")

    return df


if __name__ == "__main__":
    dataset_set = ["navjordj/SNL_summarization",
                   "jkorsvik/cnn_daily_mail_nor_final"]
    resultsdf = pd.DataFrame(
        columns=['dataset', 'model', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum' 'gen_len_words'])

    df = main(dataset_set[0])
    lead3scores = rouge_scores(df['Lead_3_summary'], df['ingress'])
    lead3scores['dataset'] = dataset_set[0]
    lead3scores['model'] = 'Lead_3'
    resultsdf = resultsdf._append(lead3scores, ignore_index=True)

    textrankscores = rouge_scores(df['TextRank_summary'], df['ingress'])
    textrankscores['dataset'] = dataset_set[0]
    textrankscores['model'] = 'TextRank_3'
    resultsdf = resultsdf._append(textrankscores, ignore_index=True)

    df = main(dataset_set[1])
    lead3scores = rouge_scores(df['Lead_3_summary'], df['highlights'])
    lead3scores['dataset'] = dataset_set[1]
    lead3scores['model'] = 'Lead_3'
    resultsdf = resultsdf._append(lead3scores, ignore_index=True)

    textrankscores = rouge_scores(df['TextRank_summary'], df['highlights'])
    textrankscores['dataset'] = dataset_set[1]
    textrankscores['model'] = 'TextRank_3'
    resultsdf = resultsdf._append(textrankscores, ignore_index=True)

    resultsdf.to_csv('baseline_results.csv', index=False)
