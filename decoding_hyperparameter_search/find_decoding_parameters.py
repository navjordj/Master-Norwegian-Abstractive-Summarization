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
    Seq2SeqTrainingArguments
)

from datasets import load_dataset
import numpy as np



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main():

    with open("space.yaml", 'r') as file:
        yaml_content = file.read()
        parsed_yaml = yaml.safe_load(yaml_content)

    n_samples = parsed_yaml.get('n_samples', None)
    validation_set_name = parsed_yaml["dataset"]
    model_name = parsed_yaml["model_name"]
    eval_batch_size = parsed_yaml["eval_batch_size"]
    fp16 = parsed_yaml.get("fp16", False)


    base_config = dict(ChainMap(*parsed_yaml["base_config"]))

    # Add max_length to all configs and create a list of dictionaries
    config_dicts = []
    for key, value in parsed_yaml['configs'].items():
        config_dict = {'name': key}
        
        # TODO: Skriv om
        config = base_config.copy()
        if value:
            for spec in value:
                key, val = spec.popitem()
                config[key] = val

        config_dict['config'] = config
        config_dicts.append(config_dict)

    print("Configurations:")
    pprint(config_dicts)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    prefix = "oppsummer: "
    text_column = "article"
    summary_column = "ingress"

    validation_set = load_dataset(validation_set_name, split="validation")
    validation_set = validation_set.select(list(range(n_samples))) if n_samples else validation_set
    print(validation_set)

    max_eval_samples = len(validation_set)

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=512, padding=False, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=512, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


    print("Preprocessing validation_set dataset")
    validation_set = validation_set.map(
        preprocess_function,
        batched=True,
        load_from_cache_file= True,
        desc="Running tokenizer on validation dataset",
    )

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    args=Seq2SeqTrainingArguments(
        output_dir = "output",
        do_eval = True,
        per_device_eval_batch_size = eval_batch_size,
        predict_with_generate=True,
        fp16 = fp16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        eval_dataset=validation_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results_df = pd.DataFrame(columns=["name", "config", 'eval_gen_len', 'eval_rouge1','eval_rouge2', 'eval_rougeL', 'eval_rougeLsum', 'eval_samples_per_second'])

    for config in tqdm.tqdm(config_dicts):
        print(config)
        gen_parameters = config["config"]
        config_name = config["name"]

        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_parameters)
        print(metrics)

        results_df = results_df.append({"name": config_name, "config": config["config"], **metrics}, ignore_index=True)
    
# Convert the DataFrame to a string
    results_df.to_csv(rf"{validation_set_name}_{model_name}_results.csv".replace("/", "_"))

    print(results_df)

if __name__ == "__main__":
    main()