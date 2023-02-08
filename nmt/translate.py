DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_CONFIG = {}

TRANSLATION_PREFIX = ">>nob<< "

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 1000

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List, Dict

import os
import time
import torch
from tqdm import tqdm

import regex as re

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(NMT_MODEL_NAME, )
model = AutoModelForSeq2SeqLM.from_pretrained(NMT_MODEL_NAME)
print(device)
dataset_train = load_dataset(DATASET_NAME, CONFIG, split='train')
dataset_test = load_dataset(DATASET_NAME, CONFIG, split='test')
dataset_valid = load_dataset(DATASET_NAME, CONFIG, split='validation')

model.to(device)

def clean_up_example(example):
    #Pre-processing
    # Append >>nob<< token
    example["article"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["article"])] + ["<EOA>"]#.append("<article-end>")
    example["highlights"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["highlights"])] + ["<EOA>"] #.append("<highlights-end>")
    return example




pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=device)
batch_size = 8

from transformers.pipelines.pt_utils import KeyDataset

dataset_dict = {"train": {"length": 287113, "data": None}, "valid": {"length": 13368, "data": None}, "test": {"length": 11490, "data": None}}
dataset_mapping = {"test": dataset_test, "train": dataset_train, "valid": dataset_valid}
for name, dataset in dataset_mapping.items():
    translations_dict = {"article": [], "highlights": []}
    #dataset.set_format(type='torch')
    dataset = dataset.map(clean_up_example)
    generated_text = []
    for col in translations_dict.keys():
        for i, out in enumerate(tqdm(pipe(KeyDataset(dataset, col),batch_size=batch_size), total=dataset_dict[name]["length"])):#/batch_size)):
            print(out)
            generated_text.append(out[0]["generated_text"])

        with torch.no_grad():
            torch.cuda.empty_cache()
        translations_dict[col] = generated_text
    dataset_dict[name]["data"] = translations_dict
 
import joblib
joblib.dump(dataset_dict, "dataset_dict.gz")

print(translations_dict[col])
