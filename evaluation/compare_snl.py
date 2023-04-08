# get CNN predictions on test set from link
import pandas as pd
import datasets
from nltk.tokenize import sent_tokenize
import nltk
from thefuzz import fuzz
from thefuzz import process
from difflib import SequenceMatcher as SM

import tqdm
nltk.download('punkt')

snl_test = datasets.load_dataset("navjordj/SNL_summarization", split="test").to_pandas()
snl_pred = pd.read_csv("https://huggingface.co/navjordj/t5-large-snl-2/raw/main/generated_predictions.txt", sep="\t", header=None)

def ratio(s1,  s2):

    if s1 is None: raise TypeError("s1 is None")
    if s2 is None: raise TypeError("s2 is None")

    m = SM(None, s1, s2)
    return int(100 * m.ratio())

def difflib_check():
    for i in range(len(snl_test)):
        matches = []
        test_article_sentences = sent_tokenize(snl_test.iloc[i]["article"])
        test_article_sentences = [x.lower() for x in test_article_sentences]
        pred_sentences = sent_tokenize(snl_pred.iloc[i][0])
        pred_sentences = [x.lower() for x in pred_sentences]
        for j in range(len(test_article_sentences)):
            for k in range(len(pred_sentences)):
                match = ratio(test_article_sentences[j], pred_sentences[k])
                if match > 95:
                    matches.append(match)
                    print(test_article_sentences[j], pred_sentences[k])
        #print(f"Article {i} has {len(matches)} matches")
        

def ratio_fuzz(sentence, choices):

    return process.extractOne(sentence, choices)

def checkup_test_pred(dataset, pred_txt_url):
    snl_matches = []
    for i in tqdm.tqdm(range(len(snl_test))):
            matches = []
            test_article_sentences = sent_tokenize(snl_test.iloc[i]["article"])
            test_article_sentences = [x.lower() for x in test_article_sentences]
            pred_sentences = sent_tokenize(snl_pred.iloc[i][0])
            pred_sentences = [x.lower() for x in pred_sentences]
            for sentence in pred_sentences:
                match = ratio_fuzz(sentence, test_article_sentences)
                if match[1] > 95:
                    matches.append(match)
            snl_matches.append(matches)

    # CNN matches is a list of lists of tuples
    # save to csv
    snl_matches_df = pd.DataFrame(snl_matches)
    snl_matches_df.to_csv("snl_matches.csv", index=False)



                



