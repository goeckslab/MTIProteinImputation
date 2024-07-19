import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List
from statannotations.Annotator import Annotator
import logging

image_folder = Path("figures", "supplements", "spatial_supplements")


def create_bar_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List, model: str, legend_position: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "red"}
    hue = "FE"
    hue_order = microns
    ax = sns.barplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)

    ax.set_ylabel("")
    ax.set_xlabel("")

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.legend(bbox_to_anchor=legend_position, loc='center', fontsize=7, ncol=3)

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylim(ylim[0], ylim[1])

    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue

        # Create pairs of (micron, 0 µm) for each marker
        for marker in data["Marker"].unique():
            pairs.append(((marker, micron), (marker, microns[0])))

    try:

        order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21',
                 'Vimentin', 'pERK', 'EGFR', 'ER']
        annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                              hide_non_significant=True)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=2, line_height=0.01,
                            comparisons_correction="Benjamini-Hochberg")
        annotator.apply_and_annotate()

    except:
        logging.error(f"Model: {model}")
        logging.error(pairs)
        logging.error(data["FE"].unique())
        raise

    return ax


def load_scores(spatial_categories: []):
    # create a new list using the inputs
    spatial_categories_strings = [f"{spatial_category} µm" for spatial_category in spatial_categories]

    # load ae multi scores
    ae_m_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))

    # sort by markers
    # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    ae_m_scores = ae_m_scores[ae_m_scores["FE"].isin(spatial_categories)]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_m_scores = ae_m_scores[
        (ae_m_scores["Mode"] == "EXP") & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0) & (
                ae_m_scores["HP"] == 0)]

    # Add µm to the FE column
    ae_m_scores["FE"] = ae_m_scores["FE"].astype(str) + " µm"
    ae_m_scores["FE"] = pd.Categorical(ae_m_scores['FE'], spatial_categories_strings)
    # rename 23 to 15, 92 to 60 and 184 to 120
    ae_m_scores["FE"] = ae_m_scores["FE"].cat.rename_categories(spatial_categories_strings)
    # sort by marker and FE
    ae_m_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    ae_m_scores = ae_m_scores[np.abs(ae_m_scores["MAE"] - ae_m_scores["MAE"].mean()) <= (3 * ae_m_scores["MAE"].std())]
    ae_m_scores = ae_m_scores[
        np.abs(ae_m_scores["RMSE"] - ae_m_scores["RMSE"].mean()) <= (3 * ae_m_scores["RMSE"].std())]

    # assert that ae_m_scores are not empty
    assert not ae_m_scores.empty, "ae_m_scores should not be empty"
    return ae_m_scores, spatial_categories_strings


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    dpi = 300

    ae_scores, spatial_categories_strings = load_scores([0, 15, 30])

    # Create new figure
    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    gspec = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.set_title('AE M 0 µm, 15 µm and 30 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax1.text(-0.01, 1.3, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)

    ax1 = create_bar_plot(data=ae_scores, metric="MAE", ylim=[0, 0.3],
                            microns=spatial_categories_strings, model="AE", legend_position=[0.15, 0.9])

    ae_scores, spatial_categories_strings = load_scores([0, 60, 90])

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('AE M 0 µm, 60 µm and 90 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax2.text(x=-0.01, y=1.3, s="b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)
    ax2 = create_bar_plot(data=ae_scores, metric="MAE", ylim=[0, 0.3],
                            microns=spatial_categories_strings, model="AE M", legend_position=[0.15, 0.9])

    ae_scores, spatial_categories_strings = load_scores([0, 120])

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.set_title('AE M 0 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax3.text(x=-0.01, y=1.3, s="b", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)
    ax3 = create_bar_plot(data=ae_scores, metric="MAE", ylim=[0, 0.3],
                            microns=spatial_categories_strings, model="AE M", legend_position=[0.15, 0.9])

    plt.tight_layout()
    plt.savefig(Path(image_folder, "ae_m_spatial.png"), dpi=300, bbox_inches='tight')
