import yaml
from pprint import pprint

from transformers import pipeline
import transformers 
import os
from datasets import load_dataset
from evaluate import evaluator
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import tqdm

import deepspeed

import random

deepspeed.init_distributed()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
# create the model pipeline

print(f"Local rank: {local_rank}, world size: {world_size}")


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
#print(parsed_yaml)
for key, value in parsed_yaml['configs'].items():
    config_dict = {'name': key}
    
    # TODO: Skriv om
    config = {}
    if value:
        for spec in value:
            print(spec)
            key, val = spec.popitem()
            config[key] = val


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
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=local_rank, **config["config"])

    pipe.model = deepspeed.init_inference(pipe.model,
            dtype=torch.float16,
            mp_size=world_size,
            replace_with_kernel_inject=True,
            max_tokens=1024,
        )

    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=validation_set,
        input_column="article",
        label_column="ingress",
        metric=metric
    )

    results_df = results_df.append({"config": config, **results}, ignore_index=True)
    
# Convert the DataFrame to a string
results_str = results_df.to_csv(index=False)
results_tensor = torch.tensor(bytearray(results_str, 'utf-8'), dtype=torch.uint8, device=local_rank)

# Get the length of the results tensor on each GPU
tensor_lengths = [torch.tensor(len(results_tensor), dtype=torch.int64, device=local_rank) for _ in range(world_size)]
torch.distributed.all_gather(tensor_lengths, torch.tensor(len(results_tensor), dtype=torch.int64, device=local_rank))

gathered_tensors = [torch.empty(length.item(), dtype=torch.uint8, device=local_rank) for length in tensor_lengths]

#All_gather the results from each GPU
torch.distributed.all_gather(gathered_tensors, results_tensor)

#Decode the results and concatenate the DataFrames
from io import BytesIO

gathered_results = [pd.read_csv(BytesIO(tensor.cpu().numpy().tobytes())) for tensor in gathered_tensors]


#Save results to a single file if this is the main process (rank 0)
if local_rank == 0:
    final_results_df = pd.concat(gathered_results, ignore_index=True)
    final_results_df.to_csv(rf"per_process_{validation_set_name}{model_name}results.csv".replace("/", ""))
    
    averaged_results_df = final_results_df.groupby("config", as_index=False).mean()

    averaged_results_df.to_csv(rf"averaged_{random.random()}{validation_set_name}{model_name}results.csv".replace("/", ""))