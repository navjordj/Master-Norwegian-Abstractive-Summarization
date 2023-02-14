#!/usr/bin/env python

# Some code borrowed from
# https://stackoverflow.com/questions/68185061/strange-results-with-huggingface-transformermarianmt-translation-of-larger-tex

# Supress warnings
import math
import warnings
import regex as re
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from core.marian import MarianOnnx
import torch
warnings.filterwarnings("ignore")

# processing libraries

# On Premises Translation support libraries

# ONNX support libraries


# On Premises Translation support libraries

# ONNX support libraries

#from accelerate import Accelerator

DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_CONFIG = {}
BATCH_SIZE = 4

TRANSLATION_PREFIX = ">>nob<< "
PADDING_VALUE = 54775
PADDING_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKOWN_TOKEN = "<unk>"

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512


def clean_up_string(string: str) -> str:
    return [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(
            SPLIT_PATTERN, string)] + ["[<end_stop_token>]"]
# dict[str, str]) -> dict[str, list[str]]:#tuple[List[str], List[str]]:


def clean_up_example(batch: dict[str, str]) -> dict[str, list[str]]:
    # Pre-processing
    # Append >>nob<< token
    stream = []
    for article, highlights in zip(batch["article"], batch["highlights"]):
        stream.extend(clean_up_string(article))
        stream.extend(clean_up_string(highlights))

        """stream["article"].extend([TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(
            SPLIT_PATTERN, article)] + ["[<end_stop_token>]"])  # .append("<article-end>")
        stream["highlights"].extend([TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(
            SPLIT_PATTERN, highlights)] + ["[<end_stop_token>]"])  # .append("<highlights-end>")
        # example #(example["article"], example["highlights"], example["id"])"""

    return {"batch_of_sentences": stream}


class localtranslate():
    def __init__(self, modelname, modelpath="outs/en-no/en-no", languagepair="en-no", onnxpath="outs/en-no/en-no"):
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

    def translate(self, sentences: list[str]):
        "list of sentences to translate"
        print(sentences)
        # Translate the text using the on prem model
        # model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True, max_length=500).to(self.device)
        input_ids = self.tokenizer(
            sentences, return_tensors='pt', padding=True).to(self.device)
        tokens = self.model.generate(**input_ids)

        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    # create a data loader generation a sentence stream from a datasets
    # Dataloader and the dataset contains "article" and "highlights" fields
    def generate_dataset(self, split):
        dataset = load_dataset(
            DATASET_NAME, CONFIG, split=split, num_proc=4, ignore_verifications=True)
        """dataset = dataset.map(
            lambda x: {"article": x["article"], "highlights": x["highlights"]},
            batched=True,
            batch_size=BATCH_SIZE
        )"""
        dataset.set_format(type="torch", columns=["article", "highlights"])
        dataset = dataset.map(clean_up_example, batched=True, batch_size=BATCH_SIZE,
                              remove_columns=["article", "highlights", "id"])
        return dataset

    def create_sentence_stream(self, dataset):

        batches = []
        length = 0
        batch_string = START_TOKEN
        for sentence in dataset["batch_of_sentences"]:

            if len(sentence) + length >= MAX_INPUT_LENGTH:
                length = 0
                batches.append(sentence)
            else:
                batch_string += END_TOKEN + sentence
                length += len(sentence)

            if len(batches) == BATCH_SIZE:
                # print(batches)
                yield batches
                batches = []
                length = 0

    def translate_dataset(self, split):
        # Translate the dataset
        dataset = self.generate_dataset(split)
        sentence_stream = self.create_sentence_stream(dataset)
        batch_of_batches = []
        for batch in sentence_stream:
            batch_of_batches.append(batch)
            if batch_of_batches == BATCH_SIZE:
                print(self.translate(batch_of_batches))
                batch_of_batches = []

        print("--------------------------------------------------")


if __name__ == "__main__":
    translator = localtranslate(modelname=NMT_MODEL_NAME)
    translator.loadmodel()
    # Load the dataset
    # translator.translate_dataset(split="test")
    translator.translate_dataset(split="validation")
    # translator.translate_dataset(split="train")
