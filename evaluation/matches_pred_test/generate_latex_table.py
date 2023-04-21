import csv
import os
import pandas as pd
import numpy as np

# List of csv file names
csv_files = ['test/SNL_summarization_test_base_matches.csv',
             'test/SNL_summarization_test_large_matches.csv',
             'test/SNL_summarization_test_ingress_matches.csv',
             'test/cnn_daily_mail_nor_final_test_base_matches.csv',
             'test/cnn_daily_mail_nor_final_test_large_matches.csv',
             'test/cnn_daily_mail_nor_final_test_highlights_matches.csv']

# Read the csv files and store the matches in a list
matches_list = []


def get_values_from_csv(file):
    df = pd.read_csv(file)
    match_ratio = df.matches / df.n_sentences
    return match_ratio.mean(), match_ratio.std(), df.coverage.mean(), df.coverage.std(), df.density.mean(), df.density.std()


mean_match_ratios = []
std_match_ratios = []
mean_coverages = []
std_coverages = []
mean_densities = []
std_densities = []

for file in csv_files:
    mean_match_ratio, std_match_ratio, mean_coverage, std_coverage, mean_density, std_density = get_values_from_csv(
        file)
    mean_match_ratios.append(mean_match_ratio)
    std_match_ratios.append(std_match_ratio)
    mean_coverages.append(mean_coverage)
    std_coverages.append(std_coverage)
    mean_densities.append(mean_density)
    std_densities.append(std_density)

# Create a DataFrame to store the results
results = pd.DataFrame({'Model': ['t5-base-snl', 't5-large-snl', 'None', 't5-base-cnndaily', 't5-large-cnndaily', 'None'],
                        'Dataset': ['SNL', 'SNL', 'SNL', 'CNN/Daily Mail', 'CNN/Daily Mail', 'CNN/Daily Mail'],
                        'Type': ['Article-Pred', 'Article-Pred', 'Article-Ingress', 'Article-Pred', 'Article-Pred', 'Article-Highlights'],
                        'Mean Match Ratio': mean_match_ratios,
                        'Std Match Ratio': std_match_ratios,
                        'Mean Coverage': mean_coverages,
                        'Std Coverage': std_coverages,
                        'Mean Density': mean_densities,
                        'Std Density': std_densities
                        })


# Generate the LaTeX table
# Make sure the table is as wide as the page wraps long names
latex_table = results.to_latex(
    index=False, float_format="%.2f", escape=False, na_rep="-", column_format="lcccccccc")

# Save the LaTeX table to a .tex file
with open('test_pred_latex_table.tex', 'w') as f:
    f.write(latex_table)

# Print the LaTeX table
print(latex_table)
