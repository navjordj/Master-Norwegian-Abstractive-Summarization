import textwrap
import numpy as np
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter("ignore", UserWarning)


def main(
    filepaths=None,
    name_plot="densityplot",
    density_upper_lim=None,
    coverage_lower_lim=None,
    colormap="coolwarm",
    n_cols=2,
    legend_location='upper right',
    show_plot=False,
):
    # Check if filepaths is None, end if
    if filepaths is None:
        return

    print("Generating density plots...")
    print("Filepaths: ", filepaths)
    print("Name plot: ", name_plot)

    # Calculate the number of rows for the subplots
    num_rows = int(np.ceil(len(filepaths) / n_cols))

    # Create a figure and define the subplots
    fig, axes = plt.subplots(num_rows, n_cols, figsize=(
        6*n_cols, 6 * num_rows), sharex=True, sharey=True)
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
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=min_compression, vmax=max_compression)

    # Loop through the filepaths and create subplots
    for idx, filepath in enumerate(filepaths):
        # Read the CSV
        data = pd.read_csv(filepath)

        # Create a density plot using seaborn kdeplot
        sns.kdeplot(
            data=data,
            x='density',
            y='coverage',
            ax=axes[idx],
            fill=True,
            cmap=colormap,
            bw_adjust=0.5,
            levels=100,
            common_norm=True,
        )

        # Compute the mean compression, mean match ratio and mean number of sentences
        mean_compression = np.mean(data['compression'])
        match_ratio = data['matches'] / data['n_sentences']
        mean_match_ratio = np.mean(match_ratio)
        mean_n_sentences = np.mean(data['n_sentences'])

        # Create a single legend with all mean values
        legend_elements = [
            mpatches.Patch(
                color='white', label=f'Compression: {mean_compression:.2f}', edgecolor='black'),
            mpatches.Patch(
                color='white', label=f'Match Ratio: {mean_match_ratio:.2f}', edgecolor='black'),
            mpatches.Patch(
                color='white', label=f'# Sentences: {mean_n_sentences:.2f}', edgecolor='black')
        ]
        axes[idx].legend(handles=legend_elements,
                         title='Mean Values', loc=legend_location)

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
            plot_name += " predicted-article"
        if "test" in descriptive_name:
            plot_name += " test"
        if "base" in descriptive_name:
            plot_name += " base"
        if "large" in descriptive_name:
            plot_name += " large"
        if "validation" in descriptive_name:
            plot_name += " validation"
        if "train" in descriptive_name:
            plot_name += " train"
        subplot_title = " ".join(x[0].upper() + x[1:]
                                 for x in plot_name.split(" "))
        axes[idx].set_title(
            subplot_title, fontsize=12)  # , fontweight='bold')

        # Set axis labels
        axes[idx].set_xlabel('Density')
        axes[idx].set_ylabel('Coverage')
        if density_upper_lim is not None and density_upper_lim > 0 and type(density_upper_lim) == int:
            axes[idx].set_xlim(0, density_upper_lim)
        if coverage_lower_lim is not None and coverage_lower_lim > 0 and type(coverage_lower_lim) == int:
            axes[idx].set_ylim(bottom=coverage_lower_lim)

    # Remove empty subplots
    for idx in range(len(filepaths), len(axes)):
        fig.delaxes(axes[idx])

    # Add the common color bar for all subplots
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(plt.cm.ScalarMappable(
    #    norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical', aspect=40)
    # cbar.set_label('Compression')
    if "densityplot_" in name_plot:
        name_plot = name_plot.replace("densityplot_", "")
    # Set the title for the figure
    fig_title = f'Normalized Bivariate Density Plots for Coverage and Density {" ".join(x[0].upper() + x[1:] for x in name_plot.split("_"))}'
    fig_title = fig_title.replace("Cnn", "CNN Daily Mail")
    fig_title = fig_title.replace("Snl", "SNL")
    fig_title = fig_title.replace(colormap, "")
    wrapped_title = textwrap.fill(fig_title, width=60)
    fig.suptitle(fig_title, fontsize=16)

    # Optimize the layout and display the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure as a high-resolution PNG file
    plt.savefig(f"plots/{name_plot}.png", dpi=300)

    if show_plot:
        plt.show()


if __name__ == '__main__':
    # List of filepaths to the CSV files
    filepaths_val_train = [
        'matches_pred_test/validation/cnn_daily_mail_nor_final_validation__highlights_matches.csv',
        'matches_pred_test/validation/SNL_summarization_validation_ingress_matches.csv',
        'matches_pred_test/train/cnn_daily_mail_nor_final_train__highlights_matches.csv',
        'matches_pred_test/train/SNL_summarization_train_ingress_matches.csv',
    ]

    filepaths_test_pred = [
        'matches_pred_test/test/cnn_daily_mail_nor_final_test__highlights_matches.csv',
        'matches_pred_test/test/SNL_summarization_test_ingress_matches.csv',
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_base_matches.csv',
        'matches_pred_test/test/SNL_summarization_test_base_matches.csv',
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_large_matches.csv',
        'matches_pred_test/test/SNL_summarization_test_large_matches.csv',
    ]

    filepaths_test_pred_snl = [
        'matches_pred_test/test/SNL_summarization_test_ingress_matches.csv',
        'matches_pred_test/test/SNL_summarization_test_base_matches.csv',
        'matches_pred_test/test/SNL_summarization_test_large_matches.csv',
    ]

    filepaths_test_pred_cnn = [
        'matches_pred_test/test/cnn_daily_mail_nor_final_test__highlights_matches.csv',
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_base_matches.csv',
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_large_matches.csv',
    ]

    # Color maps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colormap = "coolwarm"

    # Main function calls to create figures

    main(
        filepaths=filepaths_test_pred_snl,
        name_plot=f"densityplot_test_pred_snl_{colormap}",
        density_upper_lim=18,
        legend_location='lower right',
        n_cols=1,
        colormap=colormap,
    )

    main(
        filepaths=filepaths_test_pred_cnn,
        name_plot=f"densityplot_test_pred_cnn_{colormap}",
        density_upper_lim=80,
        legend_location='lower right',
        coverage_lower_lim=0.2,
        n_cols=1,
        colormap=colormap,
    )

    main(
        filepaths=filepaths_test_pred,
        name_plot=f"densityplot_test_pred_SNL_and_CNN_{colormap}",
        density_upper_lim=None,
        legend_location='lower right',
        colormap=colormap,
    )

    main(
        filepaths=filepaths_val_train,
        name_plot=f"densityplot_validation_train_SNL_and_CNN_{colormap}",
        density_upper_lim=10,
        legend_location='lower right',
        colormap=colormap,
    )
