import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
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


def create_bar_plot_null_model(data: pd.DataFrame, metric: str) -> plt.Figure:
    hue = "Model"
    ax = sns.barplot(data=data, x="Marker", y=metric, hue=hue, hue_order=["Null", "EN"],
                     palette={"EN": "lightblue", "Null": "red"})

    # Set y axis to log scale
    # ax.set_yscale('log', base=10)

    # Optional: Set title and remove axis labels if needed
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Set y axis limits
    ax.set_ylim(0, 0.6)

    # Reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Adjust legend position
    ax.legend(bbox_to_anchor=[0.6, 0.95], loc='center', ncol=2)

    hue_order = ["Null", "EN"]
    pairs = [
        (("pRB", "Null"), ("pRB", "EN")),
        (("CD45", "Null"), ("CD45", "EN")),
        (("CK19", "Null"), ("CK19", "EN")),
        (("Ki67", "Null"), ("Ki67", "EN")),
        (("aSMA", "Null"), ("aSMA", "EN")),
        (("Ecad", "Null"), ("Ecad", "EN")),
        (("PR", "Null"), ("PR", "EN")),
        (("CK14", "Null"), ("CK14", "EN")),
        (("HER2", "Null"), ("HER2", "EN")),
        (("AR", "Null"), ("AR", "EN")),
        (("CK17", "Null"), ("CK17", "EN")),
        (("p21", "Null"), ("p21", "EN")),
        (("Vimentin", "Null"), ("Vimentin", "EN")),
        (("pERK", "Null"), ("pERK", "EN")),
        (("EGFR", "Null"), ("EGFR", "EN")),
        (("ER", "Null"), ("ER", "EN")),
        (("Mean", "Null"), ("Mean", "EN")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER', "Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


def create_boxen_plot_en_vs_lgbm(data: pd.DataFrame, metric: str) -> plt.Figure:
    ax = sns.barplot(data=data, x="Marker", y=metric, hue="Network", hue_order=["EN", "LGBM"],
                     palette={"EN": "lightblue", "LGBM": "orange"})

    # Optional: Set title and remove axis labels if needed
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Set y axis limits
    ax.set_ylim(0, 0.6)

    # Reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Adjust legend position
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    hue = "Network"
    hue_order = ["EN", "LGBM"]
    pairs = [
        (("pRB", "EN"), ("pRB", "LGBM")),
        (("CD45", "EN"), ("CD45", "LGBM")),
        (("CK19", "EN"), ("CK19", "LGBM")),
        (("Ki67", "EN"), ("Ki67", "LGBM")),
        (("aSMA", "EN"), ("aSMA", "LGBM")),
        (("Ecad", "EN"), ("Ecad", "LGBM")),
        (("PR", "EN"), ("PR", "LGBM")),
        (("CK14", "EN"), ("CK14", "LGBM")),
        (("HER2", "EN"), ("HER2", "LGBM")),
        (("AR", "EN"), ("AR", "LGBM")),
        (("CK17", "EN"), ("CK17", "LGBM")),
        (("p21", "EN"), ("p21", "LGBM")),
        (("Vimentin", "EN"), ("Vimentin", "LGBM")),
        (("pERK", "EN"), ("pERK", "LGBM")),
        (("EGFR", "EN"), ("EGFR", "LGBM")),
        (("ER", "EN"), ("ER", "LGBM")),
        (("Mean", "EN"), ("Mean", "LGBM")),
    ]
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER', "Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # load null model scores
    null_model_scores = pd.read_csv(f"results/scores/null_model/scores.csv")
    # rename Protein to Marker
    null_model_scores = null_model_scores.rename(columns={"Protein": "Marker"})
    # scale MAE scores between 0 and 1
    null_model_scores["MAE"] = (null_model_scores["MAE"] - null_model_scores["MAE"].min()) / (
            null_model_scores["MAE"].max() - null_model_scores["MAE"].min())

    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    # rename lgbm scores EXP TO AP
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})
    # select only AP scores
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "AP"]
    # sort by marker
    lgbm_scores.sort_values(by=["Marker"], inplace=True)

    # calculate mean performance for each marker and mode
    lgbm_mean = lgbm_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    lgbm_mean = lgbm_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean["Marker"] = "Mean"
    lgbm_mean["FE"] = 0
    lgbm_mean["HP"] = 1
    lgbm_mean["Network"] = "LGBM"

    # add a new row to lgbm scores, adding the mean scores
    lgbm_scores = lgbm_scores.append(lgbm_mean, ignore_index=True)

    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    # rename EXP to AP
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})
    # select only AP scores
    en_scores = en_scores[en_scores["Mode"] == "AP"]
    # sort by marker
    en_scores.sort_values(by=["Marker"], inplace=True)

    # calculate mean performance for each marker and mode
    en_mean = en_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    # calculate the mean of the mean for each mode
    en_mean = en_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean["Marker"] = "Mean"
    en_mean["FE"] = 0
    en_mean["HP"] = 0
    en_mean["Network"] = "EN"
    en_scores = en_scores.append(en_mean, ignore_index=True)

    # calculate mean performance for each marker and mode
    null_mean = null_model_scores.groupby(["Model", "Biopsy", "Marker"]).mean(numeric_only=True).reset_index()
    # calculate mean for each model
    null_mean["Marker"] = "Mean"
    null_mean["FE"] = 0
    null_mean["HP"] = 0
    null_model_scores = null_model_scores.append(null_mean, ignore_index=True)

    combined_en_lgbm_scores = pd.concat([en_scores, lgbm_scores])

    bx_data = {}
    for patient in PATIENTS:
        patient_scores: pd.DataFrame = pd.read_csv(
            Path("data", "bxs", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"), sep="\t")

        patient_scores = patient_scores.loc[(patient_scores != 0.0).any(axis=1)]
        bx_data[patient] = patient_scores

    fig = plt.figure(figsize=(10, 8), dpi=300)
    gspec = fig.add_gridspec(6, 4)

    ax11 = fig.add_subplot(gspec[2:4, :1])
    # remove box from ax1
    plt.box(False)
    ax11.text(-0.5, 1.15, "b", transform=ax11.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')

    # sns.histplot(gt, color="blue", label="Expression", kde=True)
    hist = sns.histplot(bx_data["9_2"]["CK19"], color="blue", ax=ax11, kde=True, stat="count")
    hist.set(ylabel="CK19", xlabel=" ")
    sns.histplot(bx_data["9_3"]["CK19"], color="green", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["CK19"], color="yellow", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["CK19"], color="red", ax=ax11, kde=True, stat="count")

    ax12 = fig.add_subplot(gspec[2:4, 1:2])
    # remove box from ax1
    plt.box(False)

    hist = sns.histplot(bx_data["9_2"]["ER"], color="blue", ax=ax12, kde=True, stat="count")
    hist.set(ylabel="ER", xlabel=" ")
    sns.histplot(bx_data["9_3"]["ER"], color="green", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["ER"], color="yellow", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["ER"], color="red", ax=ax12, kde=True, stat="count")

    ax13 = fig.add_subplot(gspec[2:4, 2:3])
    # remove box from ax1
    plt.box(False)

    hist = sns.histplot(bx_data["9_2"]["pRB"], color="blue", ax=ax13, kde=True, stat="count")
    hist.set(ylabel="pRB", xlabel=" ")
    sns.histplot(bx_data["9_3"]["pRB"], color="green", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["pRB"], color="yellow", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["pRB"], color="red", ax=ax13, kde=True, stat="count")

    ax14 = fig.add_subplot(gspec[2:4, 3:4])
    # remove box from ax1
    plt.box(False)

    hist = sns.histplot(bx_data["9_2"]["CK17"], color="blue", ax=ax14, kde=True, stat="count")
    hist.set(ylabel="CK17", xlabel=" ")
    sns.histplot(bx_data["9_3"]["CK17"], color="green", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["CK17"], color="yellow", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["CK17"], color="red", ax=ax14, kde=True, stat="count")
    plt.legend(labels=["9 2", "9 3", "9 14", "9 15"], loc='upper left', ncol=1, bbox_to_anchor=(0.4, 1.05))
    # create legend with custom labels

    ax1 = fig.add_subplot(gspec[0:2, :])
    ax1.text(-0.1, 1.15, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title('Null & EN MAE', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax1 = create_bar_plot_null_model(data=null_model_scores, metric="MAE")

    ax2 = fig.add_subplot(gspec[4:6, :])
    ax2.text(-0.1, 1.15, "c", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax2.set_title('EN & LGBM MAE', rotation='vertical', x=-0.1, y=0, fontsize=12)
    ax2 = create_boxen_plot_en_vs_lgbm(data=combined_en_lgbm_scores, metric="MAE")

    plt.tight_layout()

    plt.savefig(Path(image_folder, "fig2.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig2.eps"), dpi=300, bbox_inches='tight', format='eps')
    sys.exit()
