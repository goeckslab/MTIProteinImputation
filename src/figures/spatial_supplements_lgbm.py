import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, logging
from typing import List
from statannotations.Annotator import Annotator

image_folder = Path("figures", "supplements", "spatial_supplements")


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "red"}

    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)

    # Optional: Set title and remove axis labels if needed
    ax.set_ylabel("")
    ax.set_xlabel("")
    # set ylim
    ax.set_ylim(0, 0.6)

    # remove legend from fig
    ax.legend(bbox_to_anchor=[0.15, 0.9], loc='center', fontsize=7, ncol=3)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

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
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                            comparisons_correction="Benjamini-Hochberg")
        annotator.apply_and_annotate()

    except:
        logging.error(pairs)
        logging.error(data["FE"].unique())
        raise

    return ax


def load_data(spatial_categories: []):
    # create a new list using the inputs
    spatial_categories_strings = [f"{spatial_category} µm" for spatial_category in spatial_categories]

    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))

    # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin(spatial_categories)]

    # select exp scores
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]

    # only select non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]

    # Add µm to the FE column
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], spatial_categories_strings)

    # update 23 to 15, 92 to 60 and 184 to 120
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(spatial_categories_strings)

    # sort by marker and FE
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    return lgbm_scores, spatial_categories_strings


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    dpi = 300
    # Create new figure
    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    gspec = fig.add_gridspec(3, 3)

    spatial_categories = [0, 15, 30]
    lgbm_scores, spatial_categories_strings = load_data(spatial_categories)

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.set_title('LGBM 0 µm, 15 µm and 30 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax1.text(x=-0.01, y=1.3, s="a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)
    ax1 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                            microns=spatial_categories_strings)

    spatial_categories = [0, 60, 90]
    lgbm_scores, spatial_categories_strings = load_data(spatial_categories)

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('LGBM 0 µm, 60 µm and 90 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax2.text(x=-0.01, y=1.3, s="b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)
    ax2 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                            microns=spatial_categories_strings)

    plt.tight_layout()
    plt.savefig(Path(image_folder, "lgbm_spatial.png"), dpi=300, bbox_inches='tight')

    spatial_categories = [0, 120]
    lgbm_scores, spatial_categories_strings = load_data(spatial_categories)

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.set_title('LGBM 0 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax3.text(x=-0.01, y=1.3, s="c", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)
    ax3 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                            microns=spatial_categories_strings)

    plt.tight_layout()
    plt.savefig(Path(image_folder, "lgbm_spatial.png"), dpi=300, bbox_inches='tight')
