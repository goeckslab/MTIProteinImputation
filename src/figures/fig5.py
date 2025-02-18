import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, logging
from typing import List
from statannotations.Annotator import Annotator

image_folder = Path("figures", "fig5")

def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "red"}
    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)

    # Remove axis labels
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Remove the legend from the default location and reposition it
    # (The current legend call places it at [0.125, 0.9]; we leave it as is.)
    ax.legend(bbox_to_anchor=[0.125, 0.9], loc='center', fontsize=7, ncol=2)

    # Remove box around the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Build statistical pairs (comparing each FE value vs. "0 µm" for each marker)
    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue
        for marker in data["Marker"].unique():
            pairs.append(((marker, micron), (marker, "0 µm")))

    try:
        order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2',
                 'AR', 'CK17', 'p21', 'Vimentin', 'pERK', 'EGFR', 'ER']
        annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order,
                              hue=hue, hue_order=hue_order, hide_non_significant=True)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                            comparisons_correction="Benjamini-Hochberg")
        annotator.apply_and_annotate()
    except Exception as e:
        logging.error(pairs)
        logging.error(data["FE"].unique())
        raise e

    return ax

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Define spatial categories and convert them to strings with µm
    spatial_categories = [0, 30, 60]
    spatial_categories_strings = [f"{cat} µm" for cat in spatial_categories]

    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin(spatial_categories)]
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], spatial_categories_strings)
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(spatial_categories_strings)
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Load the spatial information image (Panel a)
    spatial_information_image = plt.imread(Path(image_folder, "panel_a.png"))

    dpi = 300
    # Create a new figure with a gridspec
    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    gspec = fig.add_gridspec(2, 3)

    # --- Panel a ---
    ax1 = fig.add_subplot(gspec[0, :2])
    # Remove box and ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    # Display the image
    ax1.imshow(spatial_information_image, aspect='auto')

    # --- Panel b ---
    ax2 = fig.add_subplot(gspec[1, :])
    # Create the boxen plot (this sets its own title, but we'll override it)
    ax2 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                            microns=spatial_categories_strings)

    plt.tight_layout()

    # --- Now add panel labels and title using fig.text() so they align vertically ---
    # Define a fixed x-coordinate for labels (vertical alignment)
    label_x = -0.02

    # Get positions from each axis (in figure coordinates)
    pos_a = ax1.get_position()
    pos_b = ax2.get_position()

    # Place panel label "a" for Panel a
    fig.text(label_x, pos_a.y1, "a", ha='left', va='bottom', fontsize=12)
    # Place panel label "b" for Panel b
    fig.text(label_x, pos_b.y1, "b", ha='left', va='bottom', fontsize=12)
    # Place the title for Panel b below its top, aligned with the same x coordinate.
    # Adjust the vertical offset (here, 0.03 below pos_b.y1) as needed.
    fig.text(label_x, pos_b.y1 - 0.03, "LGBM 0 µm, 30 µm and 60 µm", ha='left', va='top',
             rotation='vertical', fontsize=12)

    # Save the figure
    fig.savefig(Path(image_folder, "fig5.png"), dpi=dpi, bbox_inches='tight')
    fig.savefig(Path(image_folder, "fig5.eps"), dpi=dpi, bbox_inches='tight', format='eps')