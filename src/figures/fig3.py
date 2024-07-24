import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
from typing import List
from statannotations.Annotator import Annotator

image_folder = Path("figures", "fig3")


def create_bar_plot_ae_ae_m(data: pd.DataFrame, metric: str, ylim: List) -> plt.Figure:
    hue = "Network"
    hue_order = ["AE", "AE M"]
    ax = sns.barplot(data=data, x="Marker", y=metric, hue=hue, hue_order=hue_order,
                     palette={"AE": "grey", "AE M": "darkgrey"})

    # Optional: Set title and remove axis labels if needed
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Set y axis limits
    ax.set_ylim(ylim[0], ylim[1])

    # Customize the x-axis tick labels
    new_labels = []
    for label in ax.get_xticklabels():
        if label.get_text() == 'Mean':
            new_label = 'Mean of\n all proteins'  # Change the label text here
            new_labels.append(new_label)
            label.set_fontstyle('italic')
            label.set_fontweight('bold')
        else:
            new_labels.append(label.get_text())

    print(new_labels)

    # Set the new tick labels
    ax.set_xticklabels(new_labels)

    # Reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Adjust legend position
    ax.legend(bbox_to_anchor=[0.7, 0.9], loc='center', ncol=2)

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
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER', "Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


def create_bar_plot_by_mode_only(data: pd.DataFrame, metric: str, ylim: List) -> plt.Figure:
    hue = "Network"
    x = "Mode"
    hue_order = ["LGBM", "EN", "AE", "AE M"]
    ax = sns.barplot(data=data, x=x, y=metric, hue=hue,
                     palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey", "AE M": "darkgrey",
                              "AE ALL": "lightgrey"})

    # plt.title(title)
    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # set y ticks of fig
    plt.box(False)
    # remove legend from fig
    plt.legend(prop={"size": 7}, loc='upper center')

    pairs = [
        (("AP", "LGBM"), ("AP", "EN")),
        (("AP", "LGBM"), ("AP", "AE")),
        (("AP", "LGBM"), ("AP", "AE M")),
        (("AP", "AE"), ("AP", "AE M")),
    ]

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # load image from image folder
    ae_workflow = plt.imread(Path("figures", "fig3", "ae_workflow.png"))

    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    # replace EXP WITH AP
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})

    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    # replace EXP WITH AP
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})

    ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    ae_m_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))
    # replace EXP WITH AP
    ae_scores["Mode"] = ae_scores["Mode"].replace({"EXP": "AP"})

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_scores = ae_scores[
        (ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
    # select only non hp scores
    ae_scores = ae_scores[ae_scores["HP"] == 0]
    ae_scores.sort_values(by=["Marker"], inplace=True)

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_m_scores = ae_m_scores[
        (ae_m_scores["FE"] == 0) & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0)]
    # select only non hp scores
    ae_m_scores = ae_m_scores[ae_m_scores["HP"] == 0]
    ae_m_scores.sort_values(by=["Marker"], inplace=True)
    # replace EXP WITH AP
    ae_m_scores["Mode"] = ae_m_scores["Mode"].replace({"EXP": "AP"})

    # calculate mean performance for each marker and mode
    ae_mean = ae_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    ae_mean = ae_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    ae_mean["Marker"] = "Mean"
    ae_mean["FE"] = 0
    ae_mean["HP"] = 0
    ae_mean["Network"] = "AE"
    ae_scores = ae_scores.append(ae_mean, ignore_index=True)

    # calculate mean performance for each marker and mode
    ae_m_mean = ae_m_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    ae_m_mean = ae_m_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    ae_m_mean["Marker"] = "Mean"
    ae_m_mean["FE"] = 0
    ae_m_mean["HP"] = 0
    ae_m_mean["Network"] = "AE M"
    ae_m_scores = ae_m_scores.append(ae_m_mean, ignore_index=True)

    # assert that FE column only contains 0
    assert (lgbm_scores["FE"] == 0).all(), "FE column should only contain 0 for lgbm_scores"
    assert (ae_m_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_m_scores"
    assert (ae_scores["FE"] == 0).all(), "FE column should only contain 0 for ae_scores"

    # select only AP scores
    ae_scores = ae_scores[ae_scores["Mode"] == "AP"]
    ae_m_scores = ae_m_scores[ae_m_scores["Mode"] == "AP"]
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "AP"]
    en_scores = en_scores[en_scores["Mode"] == "AP"]

    combined_ae_scores = pd.concat([ae_scores, ae_m_scores], axis=0)

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, en_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value", "Hyper"], inplace=True)
    # rename MOde AP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    fig = plt.figure(figsize=(10, 10), dpi=300)
    gspec = fig.add_gridspec(7, 3)

    ax1 = fig.add_subplot(gspec[:3, :])
    ax1.text(-0.05, 1.3, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title("AE Workflow", rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax1.imshow(ae_workflow, aspect='auto')
    # remove y axis from ax1
    ax1.set_yticks([])
    ax1.set_xticks([])

    ax2 = fig.add_subplot(gspec[3:5, :])
    ax2.text(-0.05, 1.2, "b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('AE MAE', rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax2 = create_bar_plot_ae_ae_m(data=combined_ae_scores, metric="MAE", ylim=[0.0, 0.3])

    ax3 = fig.add_subplot(gspec[5:7, :2])
    ax3.text(-0.08, 1.2, "c", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('Performance', rotation='vertical', x=-0.08, y=0, fontsize=12)
    ax3 = create_bar_plot_by_mode_only(data=all_scores, metric="MAE", ylim=[0.0, 0.3])

    plt.tight_layout()

    # save figure
    fig.savefig(Path(image_folder, "fig3.png"), dpi=300, bbox_inches='tight')
    fig.savefig(Path(image_folder, "fig3.eps"), dpi=300, bbox_inches='tight', format='eps')

    sys.exit()
