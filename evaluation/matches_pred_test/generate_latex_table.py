import csv
import os
import pandas as pd
import numpy as np

# List of csv file names
csv_files = ['SNL_summarization_base_matches.csv', 
             'SNL_summarization_large_matches.csv',
             'SNL_summarization_ingress_matches.csv',
             'cnn_daily_mail_nor_final_base_matches.csv',
             'cnn_daily_mail_nor_final_large_matches.csv',
             'cnn_daily_mail_nor_final_highlights_matches.csv']

# Read the csv files and store the matches in a list
matches_list = []

def get_values_from_csv(file):
    df = pd.read_csv(file)
    df.match_ratio = df.matches / df.n_sentences
    return df.match_ratio.mean(), df.match_ratio.median(), df.match_ratio.max()

averages = []
medians = []
maxs = []
for file in csv_files:
    average, median, max = get_values_from_csv(file)
    averages.append(average)
    medians.append(median)
    maxs.append(max)
    
# Create a DataFrame to store the results
results = pd.DataFrame({'Model': ['t5-base-snl', 't5-large-snl', 'None', 't5-base-cnndaily', 't5-large-cnndaily', 'None'],
                        'Dataset': ['SNL', 'SNL', 'SNL', 'CNN/Daily Mail', 'CNN/Daily Mail', 'CNN/Daily Mail'],
                        'Type': ['Article-Pred', 'Article-Pred', 'Article-Ingress', 'Article-Pred', 'Article-Pred', 'Article-Highlights'],
                        'Mean': averages,
                        'Median': medians,
                        'Max': maxs})

# Generate the LaTeX table
latex_table = results.to_latex(index=False)

# Save the LaTeX table to a .tex file
with open('latex_table.tex', 'w') as f:
    f.write(latex_table)

# Print the LaTeX table
print(latex_table)