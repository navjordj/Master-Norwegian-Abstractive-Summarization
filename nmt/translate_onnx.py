#!/usr/bin/env python

# Some code borrowed from 
# https://stackoverflow.com/questions/68185061/strange-results-with-huggingface-transformermarianmt-translation-of-larger-tex

DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_CONFIG = {}
BATCH_SIZE = 1

TRANSLATION_PREFIX = ">>nob<< "
PADDING_VALUE = 54775
PADDING_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKOWN_TOKEN = "<unk>"

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512

# Supress warnings
import math
import warnings
warnings.filterwarnings("ignore")

# processing libraries
import regex as re

# On Premises Translation support libraries
from transformers import MarianMTModel, MarianTokenizer
from transformers import  MarianTokenizer
from accelerate import Accelerator

#ONNX support libraries
from core.marian import MarianOnnx

import torch

# On Premises Translation support libraries
from transformers import MarianMTModel, MarianTokenizer
from transformers import  MarianTokenizer

#ONNX support libraries
from core.marian import MarianOnnx

def clean_up_example(example: dict[str, str]) -> dict[str, list[str]]:#tuple[List[str], List[str]]:
    #Pre-processing
    # Append >>nob<< token
    example["article"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["article"])] + ["[<end_stop_token>]"]#.append("<article-end>")
    example["highlights"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["highlights"])] + ["[<end_stop_token>]"] #.append("<highlights-end>")
    return example#example #(example["article"], example["highlights"], example["id"])


class localtranslate():
    def __init__(self, modelname, modelpath="outs/en-no", languagepair="en-no", onnxpath="outs/en-no"):
        if torch.cuda.is_available():  
            self.dev = "cuda"
            print("Using CUDA")
        else:  
            self.dev = "cpu" 
            print("Using CPU")
        self.device = torch.device(self.dev)

        # Initialize variables for on prem tokenizer and model
        self.modelname = modelname
        self.modelpath = modelpath
        self.languagepair = languagepair
        self.onnxpath = onnxpath

    
    def loadmodel(self):
        # Load the model and tokenizer from the model path
        
        self.tokenizer = MarianTokenizer.from_pretrained(self.modelpath)

        # Load the model and tokenizer from the onnx path
        self.model = MarianOnnx(self.onnxpath, device=self.dev)
       
    def translate(self, sentences : list[str]):
        "list of sentences to translate"
        # Translate the text using the on prem model
        # model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True, max_length=500).to(self.device)
        input_ids = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
        tokens = self.model.generate(**input_ids)
             
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    # create a data loader generation a sentence stream from a datasets
    # Dataloader and the dataset contains "article" and "highlights" fields
    def generate_dataset(self, dataset, split):
        dataset = dataset.load_dataset(DATASET_NAME, split=split, num_proc=4, ignore_verifications=True)
        dataset = dataset.map(
            lambda x: {"article": x["article"], "highlights": x["highlights"]},
            batched=True,
            batch_size=BATCH_SIZE
        )
        dataset.set_format(type="torch", columns=["article", "highlights"])
        dataset.map(clean_up_example, batched=True, batch_size=BATCH_SIZE)
        return dataset
    
    def create_sentence_stream(self, dataset, split):
        dataset = self.generate_dataset(dataset, split)
        batches = []
        for sample in dataset:
            length = 0
            batch_string = START_TOKEN
            for sentence in sample["article"].extend(sample["highlights"]):
                if len(sentence) + length > MAX_INPUT_LENGTH:
                    length = 0 
                    batches.append(sentence)
                else:
                    batch_string += END_TOKEN + sentence
                    length += len(sentence)

                if len(batches) == BATCH_SIZE:
                    yield batches
                    batches = []
                    length = 0

    def translate_dataset(self, dataset, split):
        # Translate the dataset
        sentence_stream = self.create_sentence_stream(dataset, split)
        for batch in sentence_stream:
            yield self.translate(batch)


if __name__ == "__main__":
    translator = localtranslate(modelname=NMT_MODEL_NAME)
    translator.loadmodel()
    # Load the dataset
    for res in translator.translate_dataset(DATASET_NAME, split="test"):
        print(res)
 