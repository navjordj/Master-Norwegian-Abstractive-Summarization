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
    print("Processing dataset: ", dataset)
    print("Processing predictions: ", pred_txt_url)
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
            match_sample_str = "".join([str(x[0]) + "< SCORE : " + str(x[1])+ " > | " for x in match_sample])
            matches.append((len(pred_sentences), len(match_sample), match_sample_str))

    # matches is a list of lists of tuples (len, matches as string)
    # save to csv
    matches_df = pd.DataFrame(matches, columns=["n_sentences", "matches", "matches_sample_str"])
    if "base" in pred_txt_url:
        res_file = f"{dataset.split('/')[-1]}_base_matches.csv"
    elif "large" in pred_txt_url:
        res_file = f"{dataset.split('/')[-1]}_large_matches.csv"
    else:
        res_file = f"{dataset.split('/')[-1]}_matches.csv"
    
    matches_df.to_csv(res_file, index=False)

def checkup_test_article_highlights(dataset, sensitivy=95):
    print("Processing dataset: ", dataset)
    test = datasets.load_dataset(dataset, split="test").to_pandas()
    matches = []
    for i in tqdm.tqdm(range(len(test))):
            match_sample = []
            test_article_sentences = sent_tokenize(test.iloc[i]["article"])
            test_article_sentences = [x.lower() for x in test_article_sentences]
            test_highlights_sentences = sent_tokenize(test.iloc[i]["highlights"])
            test_highlights_sentences = [x.lower() for x in test_highlights_sentences]
            for sentence in test_highlights_sentences:
                match = ratio_fuzz(sentence, test_article_sentences)
                if match[1] > sensitivy:
                    match_sample.append(match)
            # unpack matches to string
            match_sample_str = "".join([str(x[0]) + "< SCORE : " + str(x[1])+ " > | " for x in match_sample])
            matches.append((len(test_highlights_sentences),len(match_sample),match_sample_str ))

    # matches is a list of lists of tuples (len, matches as string)
    # save to csv
    matches_df = pd.DataFrame(matches, columns=["n_sentences", "matches", "matches_sample_str"])
    res_file = f"{dataset.split('/')[-1]}_highlights_matches.csv"
    matches_df.to_csv(res_file, index=False)


def checkup_test_article_ingress(dataset, sensitivy=95):
    print("Processing dataset: ", dataset)
    test = datasets.load_dataset(dataset, split="test").to_pandas()
    matches = []
    for i in tqdm.tqdm(range(len(test))):
            match_sample = []
            test_article_sentences = sent_tokenize(test.iloc[i]["article"])
            test_article_sentences = [x.lower() for x in test_article_sentences]
            test_ingress_sentences = sent_tokenize(test.iloc[i]["ingress"])
            test_ingress_sentences = [x.lower() for x in test_ingress_sentences]
            for sentence in test_ingress_sentences:
                match = ratio_fuzz(sentence, test_article_sentences)
                if match[1] > sensitivy:
                    match_sample.append(match)
            # unpack matches to string
            match_sample_str = "".join([str(x[0]) + "< SCORE : " + str(x[1])+ " > | " for x in match_sample])
            matches.append((len(test_ingress_sentences), len(match_sample),match_sample_str))

    # matches is a list of lists of tuples (len, matches as string)
    # save to csv
    matches_df = pd.DataFrame(matches, columns=["n_sentences","matches", "matches_sample_str"])
    res_file = f"{dataset.split('/')[-1]}_ingress_matches.csv"
    matches_df.to_csv(res_file, index=False)


if __name__ == "__main__":
    # Online chekuo
    #checkup_test_pred("navjordj/SNL_summarization", "https://huggingface.co/navjordj/t5-large-snl-2/raw/main/generated_predictions.txt")
    #checkup_test_pred("jkorsvik/cnn_daily_mail_nor_final", "https://huggingface.co/navjordj/t5-large-cnndaily/raw/main/generated_predictions.txt")

    # Local checkup with best parameters for model generation
    checkup_test_pred("navjordj/SNL_summarization", "results/navjordj_t5-base-snl_generated_predictions.txt")
    checkup_test_pred("navjordj/SNL_summarization", "results/navjordj_t5-large-snl-2_generated_predictions.txt")
    checkup_test_pred("jkorsvik/cnn_daily_mail_nor_final", "results/navjordj_t5-base-cnndaily-2_generated_predictions.txt")
    checkup_test_pred("jkorsvik/cnn_daily_mail_nor_final", "results/navjordj_t5-large-cnndaily_generated_predictions.txt")

    # Checking how the highlights match the article in the test set
    checkup_test_article_ingress("navjordj/SNL_summarization")
    checkup_test_article_highlights("jkorsvik/cnn_daily_mail_nor_final")
    

                



