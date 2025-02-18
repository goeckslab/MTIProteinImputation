import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from typing import List
from statannotations.Annotator import Annotator

image_folder = Path("figures", "fig3")


def create_bar_plot_ae_ae_m(data: pd.DataFrame, metric: str, ylim: List) -> plt.Axes:
    hue = "Network"
    hue_order = ["AE", "AE M"]
    ax = sns.boxenplot(
        data=data, x="Marker", y=metric, hue=hue, hue_order=hue_order,
        palette={"AE": "grey", "AE M": "darkgrey"}
    )

    # Remove axis labels
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Set y axis limits
    ax.set_ylim(ylim[0], ylim[1])

    # Customize the x-axis tick labels
    new_labels = ["Mean of\nall proteins" if label.get_text() == 'Mean' else label.get_text()
                  for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)

    # Remove spines
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Place legend at the bottom center (outside the plot) with one column so each entry is on a separate row
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.3), borderaxespad=0)

    # Statistical annotations
    pairs = [
        (("pRB", "AE"), ("pRB", "AE M")),
        (("CD45", "AE"), ("CD45", "AE M")),
        (("CK19", "AE"), ("CK19", "AE M")),
        (("Ki67", "AE"), ("Ki67", "AE M")),
        (("aSMA", "AE"), ("aSMA", "AE M")),
        (("Ecad", "AE"), ("Ecad", "AE M")),
        (("PR", "AE"), ("PR", "AE M")),
        (("CK14", "AE"), ("CK14", "AE M")),
        (("HER2", "AE"), ("HER2", "AE M")),
        (("AR", "AE"), ("AR", "AE M")),
        (("CK17", "AE"), ("CK17", "AE M")),
        (("p21", "AE"), ("p21", "AE M")),
        (("Vimentin", "AE"), ("Vimentin", "AE M")),
        (("pERK", "AE"), ("pERK", "AE M")),
        (("EGFR", "AE"), ("EGFR", "AE M")),
        (("ER", "AE"), ("ER", "AE M")),
        (("Mean", "AE"), ("Mean", "AE M")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2',
             'AR', 'CK17', 'p21', 'Vimentin', 'pERK', 'EGFR', 'ER', "Mean"]
    annotator = Annotator(
        ax, pairs, data=data, x="Marker", y=metric, order=order,
        hue=hue, hue_order=hue_order, verbose=1
    )
    annotator.configure(
        test='Mann-Whitney', text_format='star', loc='outside',
        comparisons_correction="Benjamini-Hochberg"
    )
    annotator.apply_and_annotate()

    return ax


def create_bar_plot_by_mode_only(data: pd.DataFrame, metric: str, ylim: List) -> plt.Axes:
    hue = "Network"
    x = "Mode"
    hue_order = ["LGBM", "EN", "AE", "AE M"]
    ax = sns.boxenplot(
        data=data, x=x, y=metric, hue=hue,
        palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey",
                 "AE M": "darkgrey", "AE ALL": "lightgrey"}
    )
    plt.ylabel("")
    plt.xlabel("")
    plt.ylim(ylim[0], ylim[1])
    plt.box(False)
    plt.legend(prop={"size": 7}, loc='upper center')

    pairs = [
        (("AP", "LGBM"), ("AP", "EN")),
        (("AP", "LGBM"), ("AP", "AE")),
        (("AP", "LGBM"), ("AP", "AE M")),
        (("AP", "AE"), ("AP", "AE M")),
    ]

    annotator = Annotator(
        ax, pairs, data=data, x=x, y=metric, hue=hue,
        hue_order=hue_order, verbose=1
    )
    annotator.configure(
        test='Mann-Whitney', text_format='star', loc='outside',
        comparisons_correction="Benjamini-Hochberg"
    )
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Load AE workflow image
    ae_workflow = plt.imread(Path("figures", "fig3", "ae_workflow.png"))

    # ------------------
    # Score Generation
    # ------------------
    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[(lgbm_scores["FE"] == 0) & (lgbm_scores["HP"] == 0)]
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})

    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})

    ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    ae_m_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))
    ae_scores["Mode"] = ae_scores["Mode"].replace({"EXP": "AP"})
    ae_scores = ae_scores[
        (ae_scores["FE"] == 0) &
        (ae_scores["Replace Value"] == "mean") &
        (ae_scores["Noise"] == 0) &
        (ae_scores["HP"] == 0)
        ]
    ae_scores.sort_values(by=["Marker"], inplace=True)

    ae_m_scores = ae_m_scores[
        (ae_m_scores["FE"] == 0) &
        (ae_m_scores["Replace Value"] == "mean") &
        (ae_m_scores["Noise"] == 0) &
        (ae_m_scores["HP"] == 0)
        ]
    ae_m_scores.sort_values(by=["Marker"], inplace=True)
    ae_m_scores["Mode"] = ae_m_scores["Mode"].replace({"EXP": "AP"})

    # Calculate mean performance for AE
    ae_mean = ae_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    ae_mean = ae_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    ae_mean["Marker"] = "Mean"
    ae_mean["FE"] = 0
    ae_mean["HP"] = 0
    ae_mean["Network"] = "AE"
    ae_scores = ae_scores.append(ae_mean, ignore_index=True)

    # Calculate mean performance for AE M
    ae_m_mean = ae_m_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    ae_m_mean = ae_m_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    ae_m_mean["Marker"] = "Mean"
    ae_m_mean["FE"] = 0
    ae_m_mean["HP"] = 0
    ae_m_mean["Network"] = "AE M"
    ae_m_scores = ae_m_scores.append(ae_m_mean, ignore_index=True)

    # Assertions to ensure FE is 0
    assert (lgbm_scores["FE"] == 0).all(), "FE column should only contain 0 for lgbm_scores"
    assert (ae_m_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_m_scores"
    assert (ae_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_scores"

    ae_scores = ae_scores[ae_scores["Mode"] == "AP"]
    ae_m_scores = ae_m_scores[ae_m_scores["Mode"] == "AP"]
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "AP"]
    en_scores = en_scores[en_scores["Mode"] == "AP"]

    combined_ae_scores = pd.concat([ae_scores, ae_m_scores], axis=0)
    all_scores = pd.concat([lgbm_scores, en_scores, ae_scores, ae_m_scores], axis=0)
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value", "Hyper"], inplace=True)
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    # ------------------
    # Figure Creation
    # ------------------
    # Create a figure using constrained layout and a gridspec with extra spacing
    fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)
    gspec = fig.add_gridspec(7, 3, wspace=0.3, hspace=0.5)

    # Panel a: AE Workflow image
    ax1 = fig.add_subplot(gspec[:3, :])
    ax1.set_title("AE Workflow", rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax1.imshow(ae_workflow, aspect='auto')
    ax1.set_yticks([])
    ax1.set_xticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Panel b: AE MAE
    ax2 = fig.add_subplot(gspec[3:5, :])
    ax2.set_title('AE MAE', rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax2 = create_bar_plot_ae_ae_m(data=combined_ae_scores, metric="MAE", ylim=[0.0, 0.3])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Panel c: Performance by Mode
    ax3 = fig.add_subplot(gspec[5:7, :2])
    ax3.set_title('Performance', rotation='vertical', x=-0.08, y=0, fontsize=12)
    ax3 = create_bar_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.3])
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # Adjust overall margins if needed
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.07)

    # --- Now add panel labels in figure coordinates ---
    # All labels use a fixed x coordinate to align vertically.
    label_x = 0  # fixed x position
    y_offset = 0.01  # vertical offset for fine tuning
    for label, ax in zip(['a', 'b', 'c'], [ax1, ax2, ax3]):
        if label == 'a' or label == 'b':
            print(label)
            y_offset = 0.1
        pos = ax.get_position()
        fig.text(label_x, pos.y1 + y_offset, label, ha='left', va='bottom', fontsize=12)

    # Save the figure
    fig.savefig(Path(image_folder, "fig3.png"), dpi=300, bbox_inches='tight')
    fig.savefig(Path(image_folder, "fig3.eps"), dpi=300, bbox_inches='tight', format='eps')
    sys.exit()