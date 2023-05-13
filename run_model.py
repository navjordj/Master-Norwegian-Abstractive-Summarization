import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, SummarizationPipeline

import csv


configs = {
    "t5-base-snl": {
        "model_name": "navjordj/t5-base-snl",
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
    "t5-large-snl": {
        "model_name": "navjordj/t5-large-snl-2",
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
    "t5-base-cnndaily": {
        "model_name": "navjordj/t5-base-cnndaily-2",
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
    "t5-large-cnndaily": {
        "model_name": "navjordj/t5-large-cnndaily",
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
}


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model name to run (e.g., t5-large-cnndaily)")
args = parser.parse_args()

# If model_name is not provided, prompt the user to select one
if args.model_name is None:
    print("Please choose a model from the following list:")
    for idx, model in enumerate(configs.keys(), start=1):
        print(f"{idx}. {model}")

    selected_model = int(input("Enter the model number: "))
    args.model_name = list(configs.keys())[selected_model - 1]

# Check if the given model name is in the configs dictionary
if args.model_name not in configs:
    raise ValueError(f"Model name '{args.model_name}' not found in configs. Please choose a valid model: {', '.join(configs.keys())}")


# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(configs[args.model_name]["model_name"])
tokenizer = AutoTokenizer.from_pretrained(configs[args.model_name]["model_name"])

# Create the summarization pipeline
summarizer = SummarizationPipeline(model=model, tokenizer=tokenizer)

# Set the generation config
generation_config = configs[args.model_name]["config"]

inputs_and_summaries = []


while True:
    # Get the text to be summarized from user input
    print("Enter the text to be summarized (press Enter twice to finish):")

    text_to_summarize = ""
    contents = []
    while True:
        line = input()
        print(line)
        if line == 'EOS':
            print("her??")
            break
        else:
            contents.append(line)

    text_to_summarize = "oppsummer: " + "\n".join(contents)

    # Check if the user wants to quit
    if text_to_summarize.lower() == "quit":
        break

    # Summarize the input text
    summary = summarizer(text_to_summarize, **generation_config)

    # Print the summarized text
    summarized_text = summary[0]["summary_text"]
    print("Summarized text:", summarized_text)

    # Store the input and summary
    inputs_and_summaries.append((text_to_summarize, summarized_text))

with open("inputs_and_summaries.csv", "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Input Text", "Summary"])
    csv_writer.writerows(inputs_and_summaries)

print("Inputs and summaries saved to 'inputs_and_summaries.csv'")
