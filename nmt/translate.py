from argparse import Namespace

DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"
NMT_MODEL_CONFIG = {}
BATCH_SIZE = 1

TRANSLATION_PREFIX = ">>nob<< "
PADDING_VALUE = 54775
PADDING_TOKEN = "<pad>"

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List, Dict

import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import regex as re

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(NMT_MODEL_NAME, )
model = AutoModelForSeq2SeqLM.from_pretrained(NMT_MODEL_NAME)#, device=device)#, load_in_8bit=True)
#print(device)
dataset_train = load_dataset(DATASET_NAME, CONFIG, split='train')
dataset_test = load_dataset(DATASET_NAME, CONFIG, split='test')
dataset_valid = load_dataset(DATASET_NAME, CONFIG, split='validation')

model.to(device)
model.gradient_checkpointing_disable()




#pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=device)


from transformers.pipelines.pt_utils import KeyDataset
# Each batch is a dictionary of lists of inputs, namely article and highlights
# Loop through dataset and translate using the transformer model with gpu, batch_size
# 1. We don't want to translate the entire dataset at once, as it will run out of memory
# 2. We want to translate the dataset in batches, but the batch size is 1, as we want to be able to
#    translate the entire article and highlights of each example at once
# 3. We want to be able to keep track of the order of the translations, as we want to be able to
#    compare the original article and highlights with the translated versions
# 4. We want to be able to translate the entire article and highlights at once, as we want to be able to

def generator_fn(dataset, col):
    dataset = KeyDataset(dataset, col)
    for i in range(len(dataset)):
        input_ids = tokenizer(dataset[i], return_tensors="pt", padding=True).input_ids
        yield input_ids

def clean_up_example(example: Dict[str, str]) -> Dict[str, List[str]]:#tuple[List[str], List[str]]:
    #Pre-processing
    # Append >>nob<< token
    example["article"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["article"])] + ["\E\O\S"]#.append("<article-end>")
    example["highlights"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["highlights"])] + ["\E\O\S"] #.append("<highlights-end>")
    return example#example #(example["article"], example["highlights"], example["id"])


def tokenize_and_cut(batch: Dict[str, List[str]]) -> tuple[List[List[int]], List[List[int]]]:
    article, highlights = batch["article"], batch["highlights"]
    
    return tokenizer(article, return_tensors="pt", padding=True).input_ids, tokenizer(highlights, return_tensors="pt", padding=True).input_ids


def sequential_transforms(*transforms) -> List[List[int]]:
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

text_transform = sequential_transforms(clean_up_example, tokenize_and_cut)

def collate_fn(batch):
    articles, highlights = [], []
    for sample in batch:
        print(sample)
        art, high = text_transform(sample)
        articles += art
        highlights += high
    articles = pad_sequence(articles, padding_value=PADDING_VALUE)
    highlights = pad_sequence(highlights, padding_value=PADDING_VALUE,)

    return articles, highlights


def evaluate(dataset,model):
    #model.eval()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                    pin_memory=True, num_workers=2, prefetch_factor=2)
    
    res_articles = []
    res_highlights = []
    for article, highlights in dataloader:
        article = article.to(device)
        
        with torch.no_grad():
            logits = model.generate(article)
        res_articles.append(" ".join(tokenizer.batch_decode(logits, skip_special_tokens=True)))
        with torch.no_grad():
            logits = model.generate(highlights)
        res_highlights.append(" ".join(tokenizer.batch_decode(logits, skip_special_tokens=True)))
    
    return res_articles, res_highlights

        

   


dataset_dict = {"train": {"length": 287113, "data": None}, "valid": {"length": 13368, "data": None}, "test": {"length": 11490, "data": None}}
dataset_mapping = {"test": dataset_test, "train": dataset_train, "valid": dataset_valid}


for name, dataset in dataset_mapping.items():
    translations_dict = {"article": [], "highlights": []}
    #dataset.set_format(type='torch')
    #dataset = dataset.map(clean_up_example)
    #dataset.set_format(type='torch', columns=["article", "highlights", "id"])
    #dataset.
    #generated_text = []
    # we only get the first sentence of the article and highlights
    # we want to translate the entire article and highlights

    #for col in translations_dict.keys():

        #for i, out in enumerate(tqdm(pipe(generator_fn(dataset, col),batch_size=BATCH_SIZE,), total=dataset_dict[name]["length"])):#/batch_size)):
        #    generated_text.append(out["generated_text"])

       
    translations_dict["article"], translations_dict["highlights"] = evaluate(dataset, model)
    dataset_dict[name]["data"] = translations_dict
    print(translations_dict["article"][0])
    with torch.no_grad():
        torch.cuda.empty_cache()
 
import joblib
joblib.dump(dataset_dict, "dataset_dict.gz")

