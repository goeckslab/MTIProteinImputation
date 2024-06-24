import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, logging
from typing import List
from statannotations.Annotator import Annotator

# logging_path = Path("src", "figures", "fig5.log")
# logging.root.handlers = []
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
#                    handlers=[
#                        logging.FileHandler(logging_path),
#                        logging.StreamHandler()
#                    ])

image_folder = Path("figures", "fig4")


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "red"}

    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)

    # Optional: Set title and remove axis labels if needed
    ax.set_ylabel("")
    ax.set_xlabel("")

    # remove legend from fig
    ax.legend(bbox_to_anchor=[0.125, 0.9], loc='center', fontsize=7, ncol=2)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Customize this list to specify which x-ticks should have arrows
    ticks_with_arrows = ['ER', 'Ecad', 'PR', 'EGFR']

    x_labels = ax.get_xticklabels()

    for label in x_labels:
        label.set_fontsize(8)
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
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                            comparisons_correction="Benjamini-Hochberg")
        annotator.apply_and_annotate()

    except:
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

    # load image from image folder
    spatial_information_image = plt.imread(Path(image_folder, "panel_a.png"))

    dpi = 300
    cm = 1 / 2.54  # centimeters in inches
    # Create new figure
    fig = plt.figure(figsize=(18.5 * cm, 12 * cm), dpi=dpi)
    gspec = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gspec[0, :2])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-0.05, 1.1, "a", transform=ax1.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # show spatial information image
    ax1.imshow(spatial_information_image, aspect='auto')

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('LGBM 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.07, y=-0.2, fontsize=7)
    ax2.text(-0.05, 1.2, "b", transform=ax2.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)
    ax2 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                            microns=spatial_categories_strings)

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig4.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig4.eps"), dpi=300, bbox_inches='tight', format='eps')
