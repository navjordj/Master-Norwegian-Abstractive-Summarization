import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np


def main():

    # List of filepaths
    filepaths = [
        'matches_pred_test/test/cnn_daily_mail_nor_final_test__highlights_matches.csv',
        'matches_pred_test/test/SNL_summarization_test_ingress_matches.csv',  # ,
        'matches_pred_test/validation/cnn_daily_mail_nor_final_validation__highlights_matches.csv',
        'matches_pred_test/validation/SNL_summarization_validation_ingress_matches.csv',  # ,
        'matches_pred_test/train/cnn_daily_mail_nor_final_train__highlights_matches.csv',
        'matches_pred_test/train/SNL_summarization_train_ingress_matches.csv',
    ]

    # Calculate the number of rows for the subplots
    num_rows = int(np.ceil(len(filepaths) / 2))

    # Create a figure and define the subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(
        12, 6 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten axes to make it easier to iterate

    # Get the global min and max compression values for the color map
    min_compression, max_compression = None, None
    for filepath in filepaths:
        data = pd.read_csv(filepath)
        min_val, max_val = data['compression'].min(), data['compression'].max()
        if min_compression is None or min_val < min_compression:
            min_compression = min_val
        if max_compression is None or max_val > max_compression:
            max_compression = max_val

    # Create a color map
    cmap = plt.get_cmap('PuRd')
    norm = plt.Normalize(vmin=min_compression, vmax=max_compression)

    # Loop through the filepaths and create subplots
    for idx, filepath in enumerate(filepaths):
        # Read the CSV
        data = pd.read_csv(filepath)

        # Create a scatterplot using seaborn with a custom color map (e.g., 'coolwarm')
        scatter = axes[idx].scatter(
            data['density'], data['coverage'], c=data['compression'], cmap=cmap, norm=norm)

        # Compute the mean compression and create the custom legend
        mean_compression = np.mean(data['compression'])
        legend_element = mpatches.Patch(
            color='white', label=f'C={mean_compression:.2f}', edgecolor='black')
        axes[idx].legend(handles=[legend_element], title='Mean Compression')

        # Extract the descriptive name from the filepath
        descriptive_name = os.path.splitext(os.path.basename(filepath))[0]
        plot_name = ""
        # Set subplot title
        if "SNL" in descriptive_name:
            plot_name += "SNL"
        if "cnn_daily_mail" in descriptive_name:
            plot_name += "CNN Daily Mail"
        if "highlights" in descriptive_name:
            plot_name += " highlights-article"
        if "ingress" in descriptive_name:
            plot_name += " ingress-article"
        if "pred" in descriptive_name:
            plot_name += " predicted"
        if "test" in descriptive_name:
            plot_name += " test"
        if "validation" in descriptive_name:
            plot_name += " validation"
        if "train" in descriptive_name:
            plot_name += " train"
        axes[idx].set_title(f'Scatterplot for {plot_name}')

        # Set axis labels
        axes[idx].set_xlabel('Density')
        axes[idx].set_ylabel('Coverage')

    # Remove empty subplots
    for idx in range(len(filepaths), len(axes)):
        fig.delaxes(axes[idx])

    # Add the common color bar for all subplots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical', aspect=40)
    cbar.set_label('Compression')

    # Optimize the layout and display the figure
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    # Save the figure as a high-resolution PNG file
    plt.savefig('scatter_subplots.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
