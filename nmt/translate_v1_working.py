DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

NMT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"
NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_CONFIG = {}

TRANSLATION_PREFIX = ">>nob<< "
PADDING_VALUE = 54775
PADDING_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKOWN_TOKEN = "<unk>"

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512
BATCH_SIZE = 64

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict

import os
import time
import torch

import regex as re
from transformers import pipeline
dataset = load_dataset(DATASET_NAME, CONFIG, split="test")

tokenizer = AutoTokenizer.from_pretrained(NMT_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(NMT_MODEL_NAME,)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=device)
model.to(device)


def clean_up_string(string):#: str) -> str:
    return [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(
            SPLIT_PATTERN, string)]

def clean_up_example(sample):#: dict[str, str]) -> dict[str, list[str]]:
    # Pre-processing
    # Append >>nob<< token
    stream = []
    #for article, highlights in zip(batch["article"], batch["highlights"]):
    stream.extend(clean_up_string(sample["article"]))
    stream.extend("[<stop_artoken>]")
    stream.extend(clean_up_string(sample["highlights"]))
    stream.extend("[<stop_hightoken>]")
    return stream

def translate_batch(inp: str) -> List[str]:
    split_input = clean_up_string(inp)
    #split_input = inp
   
    
    inp_batches = [split_input[i:i + BATCH_SIZE] for i in range(0, len(split_input), BATCH_SIZE)]
    #batch = inp

    translated_out = ""
    for batch  in inp_batches:
        input_tokens = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt", 
            padding="max_length", 
            max_length=MAX_INPUT_LENGTH
        )
        #input_ids = tokenizer(split_input, return_tensors="pt", padding=True).input_ids
        #input_ids = input_ids.to(device)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(device)
        outputs = model.generate(
            **input_tokens,
            use_cache=True,
            num_beams=1,
            min_length=2,
            #max_length=MAX_INPUT_LENGTH,
            no_repeat_ngram_size=3,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #outputs = pipe(split_input)["generated_text"]
        translated_out += " " + " ".join(outputs)
   

    return translated_out
    #return outputs

def translate_sample(inp: Dict) -> Dict:
    translated_article = translate_batch(inp["article"])
    translated_highlights = translate_batch(inp["highlights"])

    return {"article": translated_article, "highlights": translated_highlights, "id": inp["id"]}


articles = []
highlights = []
id = []

dataset_dict = {"train": {"length": 287113, "data": None}, "valid": {"length": 13368, "data": None}, "test": {"length": 11490, "data": None}}


for i, out in enumerate(tqdm(dataset, total=dataset_dict["test"]["length"])):
    
    temp_dict = translate_sample(out)
    articles.append(temp_dict["article"])
    highlights.append(temp_dict["highlights"])
    id.append(temp_dict["id"])
    print(temp_dict)
    if i == 10:
        break

import pandas as pd
df = pd.DataFrame({"article": articles, "highlights": highlights, "id": id})
df.to_csv("translated_test.csv", index=False)
