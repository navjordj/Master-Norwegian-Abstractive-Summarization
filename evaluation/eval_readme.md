# Evaluation

## Evaluation of the model

## Human evaluation

sampling_for_human_evaluation.py
generates text files with the generated sentences for human evaluation. The files are in the folder human_eval_samples.
The google sheet is found at this link [Google sheets](https://docs.google.com/spreadsheets/d/1GcTdQZS2_Yh7j5U8JxafOW2M4AG5xaduWNAG98azMgM/edit?usp=sharing)

## Automatic evaluation

The automatic evaluation is done using the script evaluation.py. The script takes the following arguments:

* --model_path: path to the model
* --data_path: path to the data

## Baseline creation

## Comparison between article and output

Finding the differences between the article and the output is done using the script compare.py.
We use thefuzz library to find the differences between the article and the output. (levenstein distance)
We are also using <https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py> to find the differences between the article and the output.
Coverage, and density
