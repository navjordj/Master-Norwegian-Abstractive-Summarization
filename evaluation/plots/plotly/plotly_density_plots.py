import textwrap
import numpy as np
import os
import pandas as pd
import warnings
import plotly.figure_factory as ff
import plotly.subplots as sp
# import plotly.graph_objs as go
import plotly.graph_objects as go

warnings.simplefilter("ignore", UserWarning)

# Not used
# As i think its easier with seaborn

# Helper function to create a Plotly 2D density plot


def create_density_plot(data, colormap):
    # , colorscale=colormap, hist_color='rgba(255, 255, 255, 0)',
    """ fig = ff.create_2d_density(
       , data['coverage'],  point_size=0
    )
    fig.update_xaxes(title_text='Density')
    fig.update_yaxes(title_text='Coverage', range=[0, 1])"""

    fig = go.Figure(go.Histogram2dContour(
        x=data['density'],
        y=data['coverage'],
        colorscale='Blues'
    ))
    return fig


def main(
    filepaths=None,
    name_plot="densityplot",
    density_upper_lim=80,
    coverage_lower_lim=0,
    colormap="coolwarm",
    n_cols=2,
    show_plot=False,
):
    if filepaths is None:
        return

    print("Generating density plots...")
    print("Filepaths: ", filepaths)
    print("Name plot: ", name_plot)

    # Calculate the number of rows for the subplots
    num_rows = int(np.ceil(len(filepaths) / n_cols))

    # Create a subplot grid
    fig = sp.make_subplots(rows=num_rows, cols=n_cols, subplot_titles=[
                           os.path.splitext(os.path.basename(fp))[0] for fp in filepaths])

    # Loop through the filepaths and create subplots
    for idx, filepath in enumerate(filepaths):
        # Read the CSV
        data = pd.read_csv(filepath)

        # Create a density plot using Plotly
        density_fig = create_density_plot(
            data, colormap=colormap)

        # Add the density plot to the subplot grid
        row, col = divmod(idx, n_cols)
        fig.add_traces(density_fig.data, rows=row+1, cols=col+1)

        # Update subplot axis labels
        fig.update_xaxes(title_text='Density', row=row+1, col=col+1)
        fig.update_yaxes(title_text='Coverage', row=row+1, col=col+1)

        # Set density upper limit if provided
        if density_upper_lim is not None and density_upper_lim > 0 and type(density_upper_lim) == int:
            fig.update_xaxes(
                range=[0, density_upper_lim], row=row+1, col=col+1)

        # Set coverage lower limit if provided
        if coverage_lower_lim is not None and coverage_lower_lim > 0 and type(coverage_lower_lim) == int:
            fig.update_yaxes(
                range=[coverage_lower_lim, 1], row=row+1, col=col+1)

    # Set the title for the figure
    fig_title = f'Normalized Bivariate Density Plots for Coverage and Density {" ".join(x[0].upper() + x[1:] for x in name_plot.split("_"))}'
    fig_title = fig_title.replace("Cnn", "CNN Daily Mail")
    fig_title = fig_title.replace("Snl", "SNL")
    fig_title = fig_title.replace(colormap, "")
    wrapped_title = textwrap.fill(fig_title, width=60)

    # Update the layout of the figure
    fig.update_layout(
        title=wrapped_title,
        coloraxis=dict(colorscale=colormap),
        showlegend=False
    )

    # Save the figure as a high-resolution PNG file
    fig.write_image(f"plots/plotly/{name_plot}.png", scale=2)

    if show_plot:
        fig.show()


if __name__ == '__main__':
    # List of filepaths to the CSV files
    filepaths_val_train = [
        'matches_pred_test/validation/cnn_daily_mail_nor_final_validation_highlights_matches.csv',
        'matches_pred_test/validation/SNL_summarization_validation_ingress_matches.csv',
        'matches_pred_test/train/cnn_daily_mail_nor_final_train_highlights_matches.csv',
        'matches_pred_test/train/SNL_summarization_train_ingress_matches.csv',
    ]

    filepaths_test_pred = [
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_highlights_matches.csv',
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
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_highlights_matches.csv',
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_base_matches.csv',
        'matches_pred_test/test/cnn_daily_mail_nor_final_test_large_matches.csv',
    ]

    # Color maps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colormap = "coolwarm"

    # Main function calls to create figures

    main(
        filepaths=filepaths_test_pred_snl,
        name_plot=f"densityplot_test_pred_snl_{colormap}",
        density_upper_lim=30,
        # legend_location='lower right',
        n_cols=1,
        colormap=colormap,
    )

    main(
        filepaths=filepaths_test_pred_cnn,
        name_plot=f"densityplot_test_pred_cnn_{colormap}",
        # density_upper_lim=80,
        # legend_location='lower right',
        # coverage_lower_lim=0.2,
        n_cols=1,
        colormap=colormap,
    )

    main(
        filepaths=filepaths_test_pred,
        name_plot=f"densityplot_test_pred_SNL_and_CNN_{colormap}",
        density_upper_lim=None,
        # legend_location='lower right',
        colormap=colormap,
    )

    main(
        filepaths=filepaths_val_train,
        name_plot=f"densityplot_validation_train_SNL_and_CNN_{colormap}",
        # density_upper_lim=10,
        # legend_location='lower right',
        colormap=colormap,
    )
