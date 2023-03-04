import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import regex as re
import torch
import time
import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
#import nltk
import argparse
DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

#NMT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"
NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_CONFIG = {}
SPLIT = "train"

TRANSLATION_PREFIX = ">>nob<< "
PADDING_VALUE = 54775
PADDING_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKOWN_TOKEN = "<unk>"

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512
BATCH_SIZE = 8


#DATA_KEY = "article"

outputdir = ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str,
                        default=DATASET_NAME, help="Which HF dataset to use.")
    parser.add_argument("-c", "--config", type=str, default=CONFIG, help="Which HF dataset config to use.")
    parser.add_argument("-s", "--split", type=str,
                        default=SPLIT, help="which split to translate.")
    parser.add_argument("-o", "--output_dir", default=outputdir, type=str,
                        help="Output directory.")
    parser.add_argument("-b", "--batchsize", default=BATCH_SIZE, type=int,
                        help="Output directory.")
    #parser.add_argument("-k", "--data_key", default=DATA_KEY, type=str, help="Output directory.")
    return parser.parse_args()


#dataset = load_dataset(DATASET_NAME, CONFIG, split="test")

tokenizer = AutoTokenizer.from_pretrained(NMT_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    NMT_MODEL_NAME,
    #load_in_8bit=True, 
    #device_map="auto"
    )
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=device)
model.to(device)



def clean_up_string(string):  # : str) -> str:
    return [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(
            SPLIT_PATTERN, string)]


def clean_up_example(sample):  # : dict[str, str]) -> dict[str, list[str]]:
    # Pre-processing
    # Append >>nob<< token
    stream = []
    # for article, highlights in zip(batch["article"], batch["highlights"]):
    stream.extend(clean_up_string(sample["article"]))
    #stream.extend(clean_up_string(sample[DATA_KEY]))

    
    stream.extend("[<stop_artoken>]")
    stream.extend(clean_up_string(sample["highlights"]))
    stream.extend("[<stop_hightoken>]")
    return stream


def translate_batch(inp: str) -> List[str]:
    split_input = clean_up_string(inp)
    #split_input = inp

    inp_batches = [split_input[i:i + BATCH_SIZE]
                   for i in range(0, len(split_input), BATCH_SIZE)]
    #batch = inp

    translated_out = ""
    for batch in inp_batches:
        input_tokens = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LENGTH
        )
        #input_ids = tokenizer(split_input, return_tensors="pt", padding=True).input_ids
        #input_ids = input_ids.to(device)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(device)
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                #use_cache=True,
                num_beams=1,
                #min_length=2,
                # max_length=MAX_INPUT_LENGTH,
                #no_repeat_ngram_size=3,
            )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #outputs = pipe(split_input)["generated_text"]
        translated_out += " " + " ".join(outputs)

    return translated_out
    # return outputs


def translate_sample(inp: Dict) -> Dict:
    translated_article = translate_batch(inp["article"])
    translated_highlights = translate_batch(inp["highlights"])

    return {"article": translated_article, "highlights": translated_highlights, "id": inp["id"]}


articles = []
highlights = []
id = []

dataset_dict = {"train": {"length": 287113, "data": None}, "validation": {
    "length": 13368, "data": None}, "test": {"length": 11490, "data": None}}


def translate_dataset(dataset, file_name):
    for i, out in enumerate(tqdm(dataset, total=dataset.num_rows)):

        temp_dict = translate_sample(out)
        articles.append(temp_dict["article"])
        highlights.append(temp_dict["highlights"])
        id.append(temp_dict["id"])
        # print(temp_dict)
        #if i == 100:
        #    break


    df = pd.DataFrame({"article": articles, "highlights": highlights, "id": id})
    df.to_csv(outputdir + file_name, index=False)

if __name__ == "__main__":
    args = parse_args()
    indexes = [0,100,200,5000, 10000, 12000,13000, 15500, 18000, 20500, 23000, 25500, 33000, 45000, 67000, 69000]
    
    file_id=-1
    indices = f"[{indexes[file_id-1]}:{indexes[file_id]}]"
    DATASET_NAME = args.dataset_name
    SPLIT = args.split + indices
    outputdir = args.output_dir
    CONFIG = args.config
    BATCH_SIZE = args.batchsize
    print(SPLIT)
    #DATA_KEY = args.data_key
    dataset = load_dataset(DATASET_NAME, CONFIG, split=SPLIT)
    translate_dataset(dataset, file_name=f"translated_{SPLIT}.csv")
