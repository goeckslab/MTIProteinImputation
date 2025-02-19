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

image_folder = Path("figures", "fig6")


def create_bar_plot_by_mode_only(data: pd.DataFrame, metric: str, ylim: List, microns: List) -> plt.Axes:
    hue = "Network"
    x = "FE"
    order = microns
    hue_order = ["LGBM", "AE", "AE M"]
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue, order=order,
                       palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey", "AE M": "darkgrey"})
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    # Initially set legend below center (we will adjust it later)
    leg = ax.legend(prop={"size": 7}, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    pairs = []
    for micron in microns:
        pairs.append(((micron, "LGBM"), (micron, "AE")))
        pairs.append(((micron, "LGBM"), (micron, "AE M")))
        pairs.append(((micron, "AE"), (micron, "AE M")))

    # Assertions for debugging
    for micron in microns:
        assert len(data[data["Network"] == "LGBM"]["FE"].unique()) == len(microns), "LGBM should have all microns"
        assert len(data[data["Network"] == "AE"]["FE"].unique()) == len(microns), "AE should have all microns"
        assert len(data[data["Network"] == "AE M"]["FE"].unique()) == len(microns), "AE M should have all microns"

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, order=order, hue=hue, hue_order=hue_order, verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()
    return ax


def create_bar_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List, model: str, legend_position: tuple,
                    ticks_with_arrows: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "red"}
    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)
    ax.set_ylabel("")
    ax.set_xlabel("")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    # Initially set legend using the provided legend_position; we will adjust after tight_layout
    leg = ax.legend(bbox_to_anchor=legend_position, loc='lower center', ncol=3)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylim(ylim[0], ylim[1])

    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue
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
    except Exception as e:
        logging.error(f"Model: {model}")
        logging.error(pairs)
        logging.error(data["FE"].unique())
        raise e
    return ax


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    spatial_categories = [0, 30, 60]
    spatial_categories_strings = [f"{spatial_category} µm" for spatial_category in spatial_categories]

    # Load AE scores
    ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    ae_scores = ae_scores[ae_scores["FE"].isin(spatial_categories)]
    ae_scores = ae_scores[(ae_scores["Mode"] == "EXP") &
                          (ae_scores["Replace Value"] == "mean") &
                          (ae_scores["Noise"] == 0) &
                          (ae_scores["HP"] == 0)]
    ae_scores["FE"] = ae_scores["FE"].astype(str) + " µm"
    ae_scores["FE"] = pd.Categorical(ae_scores['FE'], spatial_categories_strings)
    ae_scores["FE"] = ae_scores["FE"].cat.rename_categories(spatial_categories_strings)
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Load AE M scores
    ae_m_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))
    ae_m_scores = ae_m_scores[ae_m_scores["FE"].isin(spatial_categories)]
    ae_m_scores = ae_m_scores[(ae_m_scores["Mode"] == "EXP") &
                              (ae_m_scores["Replace Value"] == "mean") &
                              (ae_m_scores["Noise"] == 0) &
                              (ae_m_scores["HP"] == 0)]
    ae_m_scores["FE"] = ae_m_scores["FE"].astype(str) + " µm"
    ae_m_scores["FE"] = pd.Categorical(ae_m_scores['FE'], spatial_categories_strings)
    ae_m_scores["FE"] = ae_m_scores["FE"].cat.rename_categories(spatial_categories_strings)
    ae_m_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for AE M scores
    ae_m_scores = ae_m_scores[np.abs(ae_m_scores["MAE"] - ae_m_scores["MAE"].mean()) <= (3 * ae_m_scores["MAE"].std())]
    ae_m_scores = ae_m_scores[
        np.abs(ae_m_scores["RMSE"] - ae_m_scores["RMSE"].mean()) <= (3 * ae_m_scores["RMSE"].std())]

    # Load LGBM scores
    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin(spatial_categories)]
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], spatial_categories_strings)
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(spatial_categories_strings)
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Merge scores
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    dpi = 300
    # Create a figure that fits within A4. Here we use 8" x 7", which is smaller than 8.27" x 11.69".
    fig = plt.figure(figsize=(8, 7), dpi=dpi)
    gspec = fig.add_gridspec(3, 3)

    # Panel a: AE
    ax_a = fig.add_subplot(gspec[0, :])
    ax_a = create_bar_plot(data=ae_scores, metric="MAE", ylim=[0, 0.5],
                           microns=spatial_categories_strings, model="AE",
                           legend_position=(0.5, -0.25),
                           ticks_with_arrows=["AR", "CK14", "CK19", "ER", "Ecad", "PR", "pRB", "EGFR", "CK17", "aSMA",
                                              "p21", "Vimentin"])
    # Panel b: AE M
    ax_b = fig.add_subplot(gspec[1, :])
    ax_b = create_bar_plot(data=ae_m_scores, metric="MAE", ylim=[0, 0.5],
                           microns=spatial_categories_strings, model="AE M",
                           legend_position=(0.5, -0.25),
                           ticks_with_arrows=["AR", "CK14", "CK19", "ER", "Ecad", "PR", "pRB", "CK17", "EGFR", "aSMA"])
    # Panel c: Performance by Mode
    ax_c = fig.add_subplot(gspec[2, :])
    ax_c = create_bar_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.5],
                                        microns=spatial_categories_strings)

    plt.tight_layout()
    fig.canvas.draw()  # Force update of positions

    # Adjust legends manually after layout
    for ax in [ax_a, ax_b]:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((0.5, -0.7))  # shift further down (adjust as needed)
    leg_c = ax_c.get_legend()
    if leg_c is not None:
        leg_c.set_bbox_to_anchor((0.5, -0.5))

    # --- Add panel labels and vertical titles using fig.text() ---
    # Use a fixed x-coordinate for vertical alignment
    label_x = 0
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()

    # Place panel labels ("a", "b", "c")
    fig.text(label_x, pos_a.y1 + 0.04, "a", ha='left', va='bottom', fontsize=10)
    fig.text(label_x, pos_b.y1 + 0.04, "b", ha='left', va='bottom', fontsize=10)
    fig.text(label_x, pos_c.y1 + 0.04, "c", ha='left', va='bottom', fontsize=10)

    # Place vertical titles for each panel aligned along the same x coordinate
    fig.text(label_x - 0.003, pos_a.y1, "AE S 0 µm, 30 µm and 60 µm", ha='left', va='top',
             rotation='vertical', fontsize=10)
    fig.text(label_x - 0.003, pos_b.y1, "AE M 0 µm, 30 µm and 60 µm", ha='left', va='top',
             rotation='vertical', fontsize=10)
    fig.text(label_x - 0.003, pos_c.y1, "Performance", ha='left', va='top',
             rotation='vertical', fontsize=10)

    plt.savefig(Path(image_folder, "fig6.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig6.eps"), dpi=dpi, bbox_inches='tight', format='eps')