import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from statannotations.Annotator import Annotator

image_folder = Path("figures", "supplements", "ip_vs_ap")
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]


def create_bar_plot(data: pd.DataFrame, metric: str) -> plt.Figure:
    # select only the shared proteins
    data = data[data["Marker"].isin(SHARED_PROTEINS)]
    ax = sns.barplot(data=data, x="Marker", y=metric, hue="Mode", hue_order=["IP", "AP"],
                     palette={"IP": "gold", "AP": "lime"})

    # remove y axis label
    plt.ylabel("")
    plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(0, 0.3)
    ax.set_yticks([0.1, 0.2, 0.3])

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # set y ticks of fig
    plt.box(False)

    plt.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    hue = "Mode"
    hue_order = ["IP", "AP"]
    pairs = [
        (("pRB", "IP"), ("pRB", "AP")),
        (("CD45", "IP"), ("CD45", "AP")),
        (("CK19", "IP"), ("CK19", "AP")),
        (("Ki67", "IP"), ("Ki67", "AP")),
        (("aSMA", "IP"), ("aSMA", "AP")),
        (("Ecad", "IP"), ("Ecad", "AP")),
        (("PR", "IP"), ("PR", "AP")),
        (("CK14", "IP"), ("CK14", "AP")),
        (("HER2", "IP"), ("HER2", "AP")),
        (("AR", "IP"), ("AR", "AP")),
        (("CK17", "IP"), ("CK17", "AP")),
        (("p21", "IP"), ("p21", "AP")),
        (("Vimentin", "IP"), ("Vimentin", "AP")),
        (("pERK", "IP"), ("pERK", "AP")),
        (("EGFR", "IP"), ("EGFR", "AP")),
        (("ER", "IP"), ("ER", "AP")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    # rename lgbm scores EXP TO AP
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})
    # sort by marker
    lgbm_scores.sort_values(by=["Marker"], inplace=True)

    # calculate mean performance for each marker and mode
    mean = lgbm_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    mean = mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    mean["Marker"] = "Mean"
    mean["FE"] = 0
    mean["HP"] = 1
    mean["Network"] = "LGBM"

    # add a new row to lgbm scores, adding the mean scores
    lgbm_scores = lgbm_scores.append(mean, ignore_index=True)

    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    # rename EXP to AP
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})
    # sort by marker
    en_scores.sort_values(by=["Marker"], inplace=True)

    # calculate mean performance for each marker and mode
    mean = en_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    mean = mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    mean["Marker"] = "Mean"
    mean["FE"] = 0
    mean["HP"] = 1
    mean["Network"] = "EN"

    en_scores = en_scores.append(mean, ignore_index=True)

    ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_scores = ae_scores[(ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
    # select only non hp scores
    ae_scores = ae_scores[ae_scores["HP"] == 0]
    ae_scores.sort_values(by=["Marker"], inplace=True)
    ae_scores = ae_scores[np.abs(ae_scores["MAE"] - ae_scores["MAE"].mean()) <= (3 * ae_scores["MAE"].std())]

    ae_scores["Mode"] = ae_scores["Mode"].replace({"EXP": "AP"})

    mean = ae_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    mean = mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    mean["Marker"] = "Mean"
    mean["FE"] = 0
    mean["HP"] = 1
    mean["Network"] = "AE"

    ae_scores = ae_scores.append(mean, ignore_index=True)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    gspec = fig.add_gridspec(6, 4)

    ax1 = fig.add_subplot(gspec[0:2, :])
    ax1.text(-0.05, 1.15, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title('Elastic Net MAE', rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax1 = create_bar_plot(data=en_scores, metric="MAE")

    ax2 = fig.add_subplot(gspec[2:4, :])
    ax2.text(-0.05, 1.15, "b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('Light GBM MAE', rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax2 = create_bar_plot(data=lgbm_scores, metric="MAE")

    ax3 = fig.add_subplot(gspec[4:6, :])
    ax3.text(-0.05, 1.15, "c", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax3.set_title('AE MAE', rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax3 = create_bar_plot(data=ae_scores, metric="MAE")

    # add title
    plt.suptitle("MAE for Elastic Net, Light GBM and AE (IP vs AP)", fontsize=16)
    plt.tight_layout()

    plt.savefig(Path(image_folder, "ip_vs_ap.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "ip_vs_ap.eps"), dpi=300, bbox_inches='tight', format='eps')
    sys.exit()
