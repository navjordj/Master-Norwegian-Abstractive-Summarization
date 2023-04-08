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


def ratio_fuzz(sentence, choices):

    return process.extractOne(sentence, choices)


def checkup_test_pred(dataset, pred_txt_url, sensitivy=95):
    test = datasets.load_dataset(dataset, split="test").to_pandas()
    pred = pd.read_csv(pred_txt_url, sep="\t", header=None)
    matches = []
    for i in tqdm.tqdm(range(len(test))):
            match_sample = []
            test_article_sentences = sent_tokenize(test.iloc[i]["article"])
            test_article_sentences = [x.lower() for x in test_article_sentences]
            pred_sentences = sent_tokenize(pred.iloc[i][0])
            pred_sentences = [x.lower() for x in pred_sentences]
            for sentence in pred_sentences:
                match = ratio_fuzz(sentence, test_article_sentences)
                if match[1] > sensitivy:
                    match_sample.append(match)
            # unpack matches to string
            match_sample_str = "".join([str(x[0]) + " SCORE : " + str(x[1]) for x in match_sample])
            matches.append((len(match_sample),match_sample_str ))

    # matches is a list of lists of tuples (len, matches as string)
    # save to csv
    matches_df = pd.DataFrame(matches, columns=["matches", "matches_sample_str"])

    matches_df.to_csv(f"{dataset.split('/')[-1]}_matches.csv", index=False)


if __name__ == "__main__":
     checkup_test_pred("navjordj/SNL_summarization", "https://huggingface.co/navjordj/t5-large-snl-2/raw/main/generated_predictions.txt")
     checkup_test_pred("jkorsvik/cnn_daily_mail_nor_final", "https://huggingface.co/navjordj/t5-large-cnndaily/raw/main/generated_predictions.txt")



                



