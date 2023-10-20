import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from typing import List
from statannotations.Annotator import Annotator

image_folder = Path("figures", "fig2")
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]


def create_boxen_plot_ip_vs_exp_quartile(data: pd.DataFrame, metric: str) -> plt.Figure:
    # plot the quartiles
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=data, hue_order=["IP", "AP"],
                       palette={"IP": "lightblue", "AP": "orange"})
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())

    hue = "Mode"
    hue_order = ["IP", "AP"]
    pairs = [
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        (("Q1", "AP"), ("Q2", "AP")),
        (("Q2", "AP"), ("Q3", "AP")),
        (("Q3", "AP"), ("Q4", "AP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=data, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    return ax


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, show_legend: bool = False) -> plt.Figure:
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Mode", hue_order=["IP", "AP"],
                       palette={"IP": "lightblue", "AP": "orange"})

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
        (("Mean", "IP"), ("Mean", "AP")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER', "Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


def create_boxen_plot_ip_vs_exp(results: pd.DataFrame, metric: str, title: str):
    # plot the quartiles
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=results, hue_order=["IP", "AP"],
                       palette={"IP": "lightblue", "AP": "orange"})
    ax.set_xlabel("Quartile")

    hue = "Mode"
    hue_order = ["IP", "AP"]
    pairs = [
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        (("Q1", "AP"), ("Q2", "AP")),
        (("Q2", "AP"), ("Q3", "AP")),
        (("Q3", "AP"), ("Q4", "AP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

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
    bx_data = {}
    for patient in PATIENTS:
        patient_scores: pd.DataFrame = pd.read_csv(
            Path("data", "bxs", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"), sep="\t")

        patient_scores = patient_scores.loc[(patient_scores != 0.0).any(axis=1)]
        bx_data[patient] = patient_scores

    biopsies = {}
    for data in Path("data", "bxs").iterdir():
        if "h5ad" in str(data):
            continue

        bx = Path(data).stem.split('.')[0]

        if "9_" not in bx:
            continue

        loaded_data = pd.read_csv(Path(data))
        loaded_data = loaded_data.loc[(loaded_data != 0.0).any(axis=1)]
        loaded_data["Biopsy"] = bx
        loaded_data["Patient"] = " ".join(bx.split('_')[0:2])
        biopsies[bx] = loaded_data

    # combine biopsies based on Patient
    for patient in PATIENTS:
        patient_data = []
        for bx in biopsies.keys():
            if patient in bx:
                patient_data.append(biopsies[bx])
        biopsies[patient] = pd.concat(patient_data)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    gspec = fig.add_gridspec(6, 4)

    ax11 = fig.add_subplot(gspec[:2, :1])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax11.set_xticks([])
    # set y ticks range
    # ax11.set_ylim([-0.2, 4.5])
    ax11.text(-0.2, 1, "a", transform=ax11.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')

    # sns.histplot(gt, color="blue", label="Expression", kde=True)
    hist = sns.histplot(bx_data["9_2"]["CK19"], color="blue", ax=ax11, kde=True, stat="count")
    hist.set(ylabel="CK19")
    sns.histplot(bx_data["9_3"]["CK19"], color="green", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["CK19"], color="yellow", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["CK19"], color="red", ax=ax11, kde=True, stat="count")
    # change x axis label
    ax11.set_xlabel("CK19")

    ax12 = fig.add_subplot(gspec[:2, 1:2])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax12.set_xticks([])
    ax12.text(-0.3, 1, "b", transform=ax12.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')

    hist = sns.histplot(bx_data["9_2"]["ER"], color="blue", ax=ax12, kde=True, stat="count")
    hist.set(ylabel="ER")
    sns.histplot(bx_data["9_3"]["ER"], color="green", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["ER"], color="yellow", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["ER"], color="red", ax=ax12, kde=True, stat="count")
    # rotate x ticks of ax12
    ax12.set_xticklabels(ax12.get_xticklabels(), rotation=90)
    ax12.set_xlabel("ER")

    ax13 = fig.add_subplot(gspec[:2, 2:3])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax13.set_xticks([])
    ax13.text(-0.3, 1, "c", transform=ax13.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')

    hist = sns.histplot(bx_data["9_2"]["pRB"], color="blue", ax=ax13, kde=True, stat="count")
    hist.set(ylabel="pRB")
    sns.histplot(bx_data["9_3"]["pRB"], color="green", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["pRB"], color="yellow", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["pRB"], color="red", ax=ax13, kde=True, stat="count")
    ax13.set_xlabel("pRB")
    # rotate x ticks of ax13
    ax13.set_xticklabels(ax13.get_xticklabels(), rotation=90)


    ax14 = fig.add_subplot(gspec[:2, 3:4])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax14.set_xticks([])
    ax14.text(-0.2, 1, "d", transform=ax14.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')

    hist = sns.histplot(bx_data["9_2"]["CK17"], color="blue", ax=ax14, kde=True, stat="count")
    hist.set(ylabel="CK17")
    sns.histplot(bx_data["9_3"]["CK17"], color="green", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["CK17"], color="yellow", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["CK17"], color="red", ax=ax14, kde=True, stat="count")
    ax14.set_xlabel("Ck17")
    plt.legend(labels=["9 2", "9 3", "9 14", "9 15"], loc='upper left', ncol=1, bbox_to_anchor=(0.4, 1.05))
    # create legend with custom labels


    # rotate x ticks of ax13
    ax14.set_xticklabels(ax13.get_xticklabels(), rotation=90)
    # Set the x-axis to log scale

    ax1 = fig.add_subplot(gspec[2:4, :])
    ax1.text(-0.1, 1.15, "e", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title('Elastic Net', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax1 = create_boxen_plot(data=en_scores, metric="MAE", ylim=[0.0, 0.4])

    ax2 = fig.add_subplot(gspec[4:6, :])
    ax2.text(-0.1, 1.15, "f", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('LBGM', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax2 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0.0, 0.4], show_legend=True)

    plt.tight_layout()

    plt.savefig(Path(image_folder, "fig2.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig2.eps"), dpi=300, bbox_inches='tight', format='eps')
    sys.exit()
