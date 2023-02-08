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

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512

# Supress warnings
import math
import warnings
warnings.filterwarnings("ignore")

# On Premises Translation support libraries
from transformers import MarianMTModel, MarianTokenizer
from transformers import  MarianTokenizer

#ONNX support libraries
from core.marian import MarianOnnx

import torch

# On Premises Translation support libraries
from transformers import MarianMTModel, MarianTokenizer
from transformers import  MarianTokenizer

#ONNX support libraries
from core.marian import MarianOnnx


class localtranslate():
    def __init__(self, modelname, modelpath, languagepair, onnxpath):
        if torch.cuda.is_available():  
            self.dev = "cuda"
            print("Using CUDA")
        else:  
            self.dev = "cpu" 
            print("Using CPU")
        self.device = torch.device(self.dev)

        # Initialize empty dictionaries for on prem tokenizer and model
        self.model={}
        self.tokenizer={}

    
    def loadmodel(self):
        # Load the model and tokenizer from the model path
        
        self.tokenizer = MarianTokenizer.from_pretrained(self.modelpath)

        # Load the model and tokenizer from the onnx path
        self.model = MarianOnnx(self.onnxpath, device=self.dev)
       
    def translate(self, text_stream : list[str]):
        "list of sentences to translate"
        # Translate the text using the on prem model
        batches = math.ceil(len(sentences) / BATCH_SIZE)
        for sentence in text_stream:
                 
            
            for i in range(batches):
                sent_batch = sentences[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                # model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True, max_length=500).to(self.device)
                input_ids = self.tokenizer(sent_batch, return_tensors='pt', padding=True).to(self.device))
                tokens = self.model.generate(**input_ids)
             
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    # create a data loader generation a sentence stream from a datasets
    # Dataloader and the dataset contains "article" and "highlights" fields
    def generate_dataset(self, dataset, split):
        dataset = dataset.load_dataset(DATASET_NAME, split=split, streaming=True)
        dataset = dataset.map(
            lambda x: {"article": x["article"], "highlights": x["highlights"]},
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=dataset["train"].column_names,
        )
        dataset.set_format(type="torch", columns=["article", "highlights"])
        
        return dataset