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
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in sentences])


# Function to calculate Rouge scores


def calculate_rouge_scores(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True)
    return scores['rouge-1'], scores['rouge-2'], scores['rouge-l']


metric = evaluate.load("rouge")

# Calculate Rouge scores for the summaries


def rouge_scores(preds, labels, tokenizer=word_tokenize):
    result = metric.compute(
        predictions=preds, references=labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len_words"] = np.mean([len(tokenizer(pred)) for pred in preds])
    return result


# Function to summarize the text using Lead-3

def lead_3_summary(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return " ".join(sentences[:3])


def main(dataset_path):
    # Load the dataset
    dataset = load_dataset(dataset_path, split="test")
    # Create a DataFrame
    df = pd.DataFrame(dataset)
    # Apply TextRank to the DataFrame
    df['TextRank_Summary'] = df['Article'].apply(textrank_summary)
    # Apply Lead-3 to the DataFrame
    df['Lead_3_Summary'] = df['Article'].apply(lead_3_summary)

    # Display the DataFrame with Lead-3 summaries
    print(df)


if __name__ == "__main__":
    main("navjordj/SNL_summarization")
