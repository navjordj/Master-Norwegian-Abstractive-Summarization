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

cnn_test = datasets.load_dataset("jkorsvik/cnn_daily_mail_nor_final", split="test").to_pandas()
cnn_pred = pd.read_csv("https://huggingface.co/navjordj/t5-large-cnndaily/raw/main/generated_predictions.txt", sep="\t", header=None)

def ratio(s1,  s2):

    if s1 is None: raise TypeError("s1 is None")
    if s2 is None: raise TypeError("s2 is None")

    m = SM(None, s1, s2)
    return int(100 * m.ratio())

def difflib_check():
    for i in range(len(cnn_test)):
        matches = []
        test_article_sentences = sent_tokenize(cnn_test.iloc[i]["article"])
        test_article_sentences = [x.lower() for x in test_article_sentences]
        pred_sentences = sent_tokenize(cnn_pred.iloc[i][0])
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

cnn_matches = []
for i in tqdm.tqdm(range(len(cnn_test))):
        matches = []
        test_article_sentences = sent_tokenize(cnn_test.iloc[i]["article"])
        test_article_sentences = [x.lower() for x in test_article_sentences]
        pred_sentences = sent_tokenize(cnn_pred.iloc[i][0])
        pred_sentences = [x.lower() for x in pred_sentences]
        for sentence in pred_sentences:
            match = ratio_fuzz(sentence, test_article_sentences)
            if match[1] > 95:
                matches.append(match)
        cnn_matches.append(matches)

# CNN matches is a list of lists of tuples
# save to csv
cnn_matches_df = pd.DataFrame(cnn_matches)
cnn_matches_df.to_csv("cnn_matches.csv", index=False)



                



