import yaml
from pprint import pprint

from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Read the content from space.yaml
with open('space.yaml', 'r') as file:
    yaml_content = file.read()

parsed_yaml = yaml.safe_load(yaml_content)

n_samples = parsed_yaml.get('n_samples', None)
default_batch_size = parsed_yaml.get('default_batch_size', None)
validation_set_name = parsed_yaml["dataset"]
model_name = parsed_yaml["model_name"]


max_length = parsed_yaml['max_length']

# Add max_length to all configs and create a list of dictionaries
config_dicts = []
print(parsed_yaml)
for key, value in parsed_yaml['configs'].items():
    config_dict = {'name': key}
    
    # Handle cases where no parameters are given
    if value is not None:
        config = value[0]
    else:
        config = {}
    
    config['max_length'] = max_length

    if "batch_size" not in config:
        config["batch_size"] = default_batch_size



    config_dict['config'] = config
    config_dicts.append(config_dict)

print("Configurations:")
pprint(config_dicts)

validation_set = load_dataset(validation_set_name, split="validation")
validation_set = validation_set.select(list(range(n_samples))) if n_samples else validation_set
print(validation_set)


metric = evaluate.load("rouge")

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

task_evaluator = evaluator("summarization")

results_df = pd.DataFrame(columns=["config", 'rouge1','rouge2', 'rougeL', 'rougeLsum', 'total_time_in_seconds', 'samples_per_second'])

for config in tqdm.tqdm(config_dicts):
    print(config)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device, **config["config"])

    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=validation_set,
        input_column="article",
        label_column="ingress",
        metric=metric
    )

    results_df = results_df.append({"config": config, **results}, ignore_index=True)

results_df.to_csv(rf"{validation_set_name}_{model_name}_results.csv".replace("/", "_"))