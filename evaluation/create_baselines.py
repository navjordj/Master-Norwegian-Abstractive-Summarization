import re
import pandas as pd
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from rouge import Rouge

# Function to summarize the text using TextRank


def textrank_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in sentences])


# Apply TextRank to the DataFrame
df['TextRank_Summary'] = df['Article'].apply(textrank_summary)

# Function to calculate Rouge scores


def calculate_rouge_scores(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True)
    return scores['rouge-1'], scores['rouge-2'], scores['rouge-l']


# Calculate Rouge scores for the summaries
rouge_scores = df.apply(lambda row: calculate_rouge_scores(
    row['TextRank_Summary'], row['Highlight']), axis=1)
df['Rouge-1'], df['Rouge-2'], df['Rouge-L'] = zip(*rouge_scores)


# Function to summarize the text using Lead-3

def lead_3_summary(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return " ".join(sentences[:3])


# Apply Lead-3 to the DataFrame
df['Lead_3_Summary'] = df['Article'].apply(lead_3_summary)

# Display the DataFrame with Lead-3 summaries
print(df)
