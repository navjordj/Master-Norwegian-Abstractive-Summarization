# https://huggingface.co/docs/transformers/v4.26.1/en/generation_strategies#customize-text-generation
# https://huggingface.co/docs/transformers/v4.26.1/en/generation_strategies#customize-text-generation

# %%
from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pandas as pd

# %%
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# %%

data = load_dataset("navjordj/SNL_summarization", split="validation")# .shuffle().select(range(10))
metric = evaluate.load("rouge")


# %%
model = T5ForConditionalGeneration.from_pretrained("navjordj/t5-base-snl")
tokenizer = AutoTokenizer.from_pretrained("navjordj/t5-base-snl")

# %%
greedy_config = {
    "max_length": 120,
}

beam_config = {
    "max_length": 120,
    "num_beams": 6,
    "early_stopping": True
}

beam_no_stopping_config = {
    "max_length": 120,
    "num_beams": 6
}

beam_rep_penalty_1_2_config = {
    "max_length": 120,
    "num_beams": 6,
    "repetition_penalty ": 1.2
}

beam_rep_penalty_2_config = {
    "max_length": 120,
    "num_beams": 6,
    "repetition_penalty ": 2
}

beam_no_ngram_2 = {
    "max_length": 120,
    "num_beams": 6,
    "no_repeat_ngram_size": 2, 
    "early_stopping": True
}

beam_no_ngram_4 = {
    "max_length": 120,
    "num_beams": 6,
    "no_repeat_ngram_size": 4, 
    "early_stopping": True
}

beam_no_ngram_12 = {
    "max_length": 120,
    "num_beams": 6,
    "no_repeat_ngram_size": 12, 
    "early_stopping": True
}

sample_config = {
    "max_length": 120,
    "do_sample": True,
    "top_k": 0
}

sample_temperature_config = {
    "max_length": 120,
    "do_sample": True,
    "top_k": 0,
    "temperature": 0.7
}

top_k_config = {
    "max_length": 120,
    "do_sample": True,
    "top_k": 50,
}

top_p_config = {
    "max_length": 120,
    "do_sample": True,
    "top_p": 0.92, 
    "top_k": 0
}

configs = [greedy_config, beam_config, beam_no_stopping_config, beam_rep_penalty_1_2_config, beam_rep_penalty_2_config, beam_no_ngram_2, beam_no_ngram_4, sample_config, sample_temperature_config, top_k_config, top_p_config]

# %%
task_evaluator = evaluator("summarization")

results_df = pd.DataFrame(columns=["config", 'rouge1','rouge2', 'rougeL', 'rougeLsum', 'total_time_in_seconds', 'samples_per_second'])

for config in configs[:3]:
    print(config)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, batch_size=8, device=device, **config)

    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=data,
        input_column="article",
        label_column="ingress",
        metric=metric
    )
    print(results)

    results_df = results_df.append({"config": config, **results}, ignore_index=True)

results_df.to_csv("hyperparam_search_results.csv")

# %%



