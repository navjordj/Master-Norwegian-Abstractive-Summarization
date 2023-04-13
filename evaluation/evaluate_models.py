import evaluate
import nltk
import pandas as pd
import tqdm
import yaml
from collections import ChainMap
from pprint import pprint

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from datasets import load_dataset
import numpy as np

import warnings

import os

N_SAMPLES = None
BATCH_SIZE = 16
PREFIX = "oppsummer: "


test_configs = [
    {
        "dataset": "navjordj/SNL_summarization",
        "model_name": "navjordj/t5-base-snl",
        "article_column": "article",
        "summary_column": "ingress",
        "config_name": "beam_no_repeat_ngram_size_3",
        "config": {
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "temperature": 1,
            "encoder_no_repeat_ngram_size": 0,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
            "num_beam_groups": 1,
            "num_beams": 5,
            "max_length": 150,
        },
    },
    {
        "dataset": "navjordj/SNL_summarization",
        "model_name": "navjordj/t5-large-snl-2",
        "article_column": "article",
        "summary_column": "ingress",
        "config_name": "beam_no_repeat_ngram_size_3",
        "config": {
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "temperature": 1,
            "encoder_no_repeat_ngram_size": 0,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
            "num_beam_groups": 1,
            "num_beams": 5,
            "max_length": 150,
        },
    },
    {
        "dataset": "jkorsvik/cnn_daily_mail_nor_final",
        "model_name": "navjordj/t5-base-cnndaily-2",
        "article_column": "article",
        "summary_column": "highlights",
        "config_name": "beam_no_repeat_ngram_size_3",
        "config": {
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "temperature": 1,
            "encoder_no_repeat_ngram_size": 0,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
            "num_beam_groups": 1,
            "num_beams": 5,
            "max_length": 150,
        },
    },
    {
        "dataset": "jkorsvik/cnn_daily_mail_nor_final",
        "model_name": "navjordj/t5-large-cnndaily",
        "article_column": "article",
        "summary_column": "highlights",
        "config_name": "beam_no_repeat_ngram_size_5",
        "config": {
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "temperature": 1,
            "encoder_no_repeat_ngram_size": 0,
            "no_repeat_ngram_size": 5,
            "do_sample": False,
            "num_beam_groups": 1,
            "num_beams": 5,
            "max_length": 150,
        },
    },
]


def evaluate_model(dataset, model_name, article_column, summary_column, config_name, config):
    print("Evaluating model: ", model_name)




    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    test_set = load_dataset(dataset, split="test")
    test_set = test_set.select(list(range(N_SAMPLES))) if N_SAMPLES else test_set
    print(test_set)

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[article_column])):
            if examples[article_column][i] and examples[summary_column][i]:
                inputs.append(examples[article_column][i])
                targets.append(examples[summary_column][i])

        inputs = [PREFIX + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=512, padding=False, truncation=True)

        labels = tokenizer(
            text_target=targets, max_length=512, padding=False, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    print("Preprocessing test_set dataset")
    test_set = test_set.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on test dataset",
    )

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):

        per_sample_scores = pd.DataFrame(columns = [
            "summary",
            "correct",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ])

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # Calculate ROUGE scores for each sample
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)

            for pred, label in zip(decoded_preds, decoded_labels):
                result = metric.compute(
                    predictions=[pred], references=[label], use_stemmer=True
                )
                result = {k: round(v * 100, 4) for k, v in result.items()}


                per_sample_scores = per_sample_scores.append({"summary": pred.strip(), "correct": label.strip(), **result}, ignore_index=True)

        per_sample_scores.to_csv(f"{model_name}_per_sample_scores.csv".replace("/", "_"))

        # Calculate ROUGE scores for the whole dataset
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)


        #Skrive prediksjoner og scorer til fil
        #decoded preds er BATCH SIZEx150
        
        return result

    args = Seq2SeqTrainingArguments(
        output_dir="output",
        do_eval=True,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

      
    predict_results = trainer.predict(test_set, metric_key_prefix="test", **config)
    """
    predictions = tokenizer.batch_decode(
        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = f"{model_name}_generated_predictions.txt".replace("/", "_")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))
    """

    return predict_results.metrics


def main():
    results_df = pd.DataFrame(
        columns=[
            "model",
            "config",
            "test_loss",
            "test_rouge1",
            "test_rouge2",
            "test_rougeL",
            "test_rougeLsum",
            "test_gen_len",
            "test_runtime",
            "test_samples_per_second",
            "test_steps_per_second",
        ]
    )

    for test_config in test_configs:


        results = evaluate_model(**test_config)
        results_df = results_df.append({"model": test_config["model_name"], "config_name": test_config["config_name"], "config": test_config["config"], **results}, ignore_index=True)
    # Convert the DataFrame to a string
    results_df.to_csv("results.csv")

    print(results_df)


if __name__ == "__main__":
    main()
