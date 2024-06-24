import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from typing import List
from statannotations.Annotator import Annotator
import logging

# logging_path = Path("src", "figures", "fig6.log")
# logging.root.handlers = []
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
#                    handlers=[
#                        logging.FileHandler(logging_path),
#                        logging.StreamHandler()
#                    ])

image_folder = Path("figures", "fig5")


def create_boxen_plot_by_mode_only(data: pd.DataFrame, metric: str, ylim: List, microns: List) -> plt.Figure:
    hue = "Network"
    x = "FE"
    order = microns
    hue_order = ["LGBM", "AE", "AE M"]
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue, order=order,
                       palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey", "AE M": "darkgrey"})

    # plt.title(title)
    # remove y axis label
    ax.set_ylabel("")
    ax.set_xlabel("")
    # plt.legend(loc='upper center')
    ax.set_ylim(ylim[0], ylim[1])

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # remove legend from fig
    ax.legend(prop={"size": 7}, loc='center', bbox_to_anchor=[0.3, 0.7])
    # plt.legend().set_visible(False)

    pairs = []
    for micron in microns:
        pairs.append(((micron, "LGBM"), (micron, "AE")))
        pairs.append(((micron, "LGBM"), (micron, "AE M")))
        pairs.append(((micron, "AE"), (micron, "AE M")))

    for micron in microns:
        # assert that for lgbm the microns are available
        assert len(data[data["Network"] == "LGBM"]["FE"].unique()) == len(microns), "LGBM should have all microns"
        # assert that for ae the microns are available
        assert len(data[data["Network"] == "AE"]["FE"].unique()) == len(microns), "AE should have all microns"
        # assert that for ae m the microns are available
        assert len(data[data["Network"] == "AE M"]["FE"].unique()) == len(microns), "AE M should have all microns"

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List, model: str, legend_position: List,
                      ticks_with_arrows: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "red"}
    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette, showfliers=False)

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

    # Add arrows below specific x-axis ticks

    x_labels = ax.get_xticklabels()

    for label in x_labels:
        label.set_fontsize(10)
        if label.get_text() in ticks_with_arrows:
            label.set_fontweight('bold')
            # Get the position of the label
            x_pos = label.get_position()[0]
            y_pos = label.get_position()[1]

            if len(label.get_text()) == 2:
                # Draw the underline
                ax.text(x_pos, y_pos - 0.06, '___', fontsize=label.get_fontsize() * 1.2, ha='center', va='top',
                        transform=ax.get_xaxis_transform())
            elif len(label.get_text()) == 3:
                # Draw the underline
                ax.text(x_pos, y_pos - 0.06, '____', fontsize=label.get_fontsize() * 1.2, ha='center', va='top',
                        transform=ax.get_xaxis_transform())
            else:
                # Draw the underline
                ax.text(x_pos, y_pos - 0.06, '_____', fontsize=label.get_fontsize() * 1.2, ha='center', va='top',
                        transform=ax.get_xaxis_transform())



    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue

        # Create pairs of (micron, 0 µm) for each marker
        for marker in data["Marker"].unique():
            pairs.append(((marker, micron), (marker, "0 µm")))

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


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # if logging_path.exists():
    #    os.remove(logging_path)

    spatial_categories = [0, 30, 60]
    # create a new list using the inputs
    spatial_categories_strings = [f"{spatial_category} µm" for spatial_category in spatial_categories]

    # load ae scores
    ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))

    # sort by markers
    # select only the scores for the 0 µm, 15 µm, 6ß µm, 120 µm
    ae_scores = ae_scores[ae_scores["FE"].isin(spatial_categories)]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_scores = ae_scores[
        (ae_scores["Mode"] == "EXP") & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0) & (
                ae_scores["HP"] == 0)]

    # Add µm to the FE column
    ae_scores["FE"] = ae_scores["FE"].astype(str) + " µm"
    ae_scores["FE"] = pd.Categorical(ae_scores['FE'], spatial_categories_strings)

    # rename 23 to 15, 92 to 60 and 184 to 120
    ae_scores["FE"] = ae_scores["FE"].cat.rename_categories(spatial_categories_strings)
    # sort by marker and FE
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # assert that ae_scores are not empty
    assert not ae_scores.empty, "ae_scores should not be empty"

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

    # rename 23 to 15, 92 to 60 and 184 to 120
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(spatial_categories_strings)

    # sort by marker and FE
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # assert that lgbm scores are not empty
    assert not lgbm_scores.empty, "lgbm_scores should not be empty"

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)
    # rename MOde EXP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    dpi = 300
    # Create new figure
    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    gspec = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.set_title('AE S 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax1.text(-0.01, 1.3, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)

    ax1 = create_boxen_plot(data=ae_scores, metric="MAE", ylim=[0, 0.6],
                            microns=spatial_categories_strings, model="AE", legend_position=[0.15, 0.9],
                            ticks_with_arrows=["AR", "CK14", "CK19", "ER", "Ecad", "PR", "pRB"])

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('AE M 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax2.text(x=-0.01, y=1.3, s="b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    # remove box from ax4
    plt.box(False)
    ax2 = create_boxen_plot(data=ae_m_scores, metric="MAE", ylim=[0, 0.6],
                            microns=spatial_categories_strings, model="AE M", legend_position=[0.15, 0.9],
                            ticks_with_arrows=["AR", "CK14", "CK19", "ER", "Ecad", "PR", "pRB"])

    ax3 = fig.add_subplot(gspec[2, :])
    ax3.text(x=-0.01, y=1.3, s="c", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('Performance', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax3 = create_boxen_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.8],
                                         microns=spatial_categories_strings)

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig5.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig5.eps"), dpi=300, bbox_inches='tight', format='eps')
