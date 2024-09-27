import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from statannotations.Annotator import Annotator

warnings.simplefilter(action='ignore', category=FutureWarning)

image_folder = Path("figures", "fig2")
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']
phenotype_folder = Path("results", "phenotypes")


# Function for creating the bar plot for Null vs EN models
def create_bar_plot_null_model(data: pd.DataFrame, metric: str, ax=None) -> plt.Figure:
    hue = "Model"
    ax = sns.barplot(data=data, x="Marker", y=metric, hue=hue, hue_order=["Null", "EN"],
                     palette={"EN": "lightblue", "Null": "red"}, ax=ax)

    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Protein")
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    for label in ax.get_xticklabels():
        if label.get_text() == 'Mean':
            label.set_fontstyle('italic')
            label.set_fontweight('bold')

    ax.set_xticklabels(
        ['Mean\nof all\nproteins' if x.get_text() == 'Mean' else x.get_text() for x in ax.get_xticklabels()])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add statistical annotations
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
        (("Mean", "Null"), ("Mean", "EN"))
    ]

    order = SHARED_PROTEINS + ["Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=["Null", "EN"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


# Function for creating the bar plot for EN vs LGBM models
def create_bar_plot_en_vs_lgbm(data: pd.DataFrame, metric: str, ax=None) -> plt.Figure:
    ax = sns.barplot(data=data, x="Marker", y=metric, hue="Network", hue_order=["EN", "LGBM"],
                     palette={"EN": "lightblue", "LGBM": "orange"}, ax=ax)

    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Protein")
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    for label in ax.get_xticklabels():
        if label.get_text() == 'Mean':
            label.set_fontstyle('italic')
            label.set_fontweight('bold')

    ax.set_xticklabels(
        ['Mean\nof all\nproteins' if x.get_text() == 'Mean' else x.get_text() for x in ax.get_xticklabels()])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add statistical annotations
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
        (("Mean", "EN"), ("Mean", "LGBM"))
    ]
    order = SHARED_PROTEINS + ["Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue="Network",
                          hue_order=["EN", "LGBM"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


# Function to plot ARI
def plot_ari():
    results = pd.read_csv("results/evaluation/cluster_metrics.csv")
    ax = sns.barplot(data=results, x="Marker", y="ARI", palette="Set2")
    ax.set_ylabel("Expression ARI Score")
    ax.set_xlabel("Protein")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    return ax


def plot_phenotype_ari(ari_scores: pd.DataFrame):
    ax = sns.barplot(data=ari_scores, x="Protein", y="Score", palette="Set2")
    ax.set_ylabel("Phenotype ARI Score")
    ax.set_xlabel("Protein")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def plot_phenotype_jaccard(jaccard_scores: pd.DataFrame):
    hue_order = ["Original CV Score", "Imputed CV Score"]
    ax = sns.barplot(data=jaccard_scores, x="Protein", y="Score", hue_order=hue_order)
    ax.set_ylabel("Phenotype Classifier Accuracy")
    ax.set_xlabel("Protein")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # change legend handles to Origin and Imputed
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:2], labels=["Original", "Imputed"], loc="lower center", ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def plot_silhouette():
    results = pd.read_csv("results/evaluation/cluster_metrics.csv")
    # create 30 replicates of the data
    results = pd.concat([results] * 30, ignore_index=True)

    melt = results.melt(id_vars=["Biopsy", "Marker"],
                        value_vars=["Silhouette Original", "Silhouette Imputed"],
                        var_name="Silhouette Type", value_name="Score")

    # rename value Silhouette Original to Original and Silhoutte Imputed to Imputed
    melt["Silhouette Type"] = melt["Silhouette Type"].replace(
        {"Silhouette Original": "Original", "Silhouette Imputed": "Imputed"})
    ax = sns.barplot(data=melt, x="Marker", y="Score", hue="Silhouette Type")
    ax.set_ylabel("Expression Silhouette Score")
    ax.set_xlabel("Protein")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # adjust legend
    ax.legend(bbox_to_anchor=[0.4, 0.97], loc='center', ncol=2, fontsize=8)

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # Add statistical annotations
    pairs = [
        (("pRB", "Original"), ("pRB", "Imputed")),
        (("CD45", "Original"), ("CD45", "Imputed")),
        (("CK19", "Original"), ("CK19", "Imputed")),
        (("Ki67", "Original"), ("Ki67", "Imputed")),
        (("aSMA", "Original"), ("aSMA", "Imputed")),
        (("Ecad", "Original"), ("Ecad", "Imputed")),
        (("PR", "Original"), ("PR", "Imputed")),
        (("CK14", "Original"), ("CK14", "Imputed")),
        (("HER2", "Original"), ("HER2", "Imputed")),
        (("AR", "Original"), ("AR", "Imputed")),
        (("CK17", "Original"), ("CK17", "Imputed")),
        (("p21", "Original"), ("p21", "Imputed")),
        (("Vimentin", "Original"), ("Vimentin", "Imputed")),
        (("pERK", "Original"), ("pERK", "Imputed")),
        (("EGFR", "Original"), ("EGFR", "Imputed")),
        (("ER", "Original"), ("ER", "Imputed")),
    ]

    order = SHARED_PROTEINS
    annotator = Annotator(ax, pairs, data=melt, x="Marker", y="Score", order=order, hue="Silhouette Type",
                          hue_order=["Original", "Imputed"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Load and process Null model scores
    null_model_scores = pd.read_csv(f"results/scores/null_model/scores.csv")
    null_model_scores = null_model_scores.rename(columns={"Protein": "Marker"})
    null_model_scores["MAE"] = (null_model_scores["MAE"] - null_model_scores["MAE"].min()) / (
            null_model_scores["MAE"].max() - null_model_scores["MAE"].min())

    # Load LGBM model scores
    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "AP"]
    lgbm_scores.sort_values(by=["Marker"], inplace=True)

    # Calculate the mean performance for each marker and mode
    lgbm_mean = lgbm_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean = lgbm_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean["Marker"] = "Mean"
    lgbm_mean["FE"] = 0
    lgbm_mean["HP"] = 1
    lgbm_mean["Network"] = "LGBM"
    lgbm_scores = pd.concat([lgbm_scores, lgbm_mean], ignore_index=True)

    # Load EN model scores
    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})
    en_scores = en_scores[en_scores["Mode"] == "AP"]
    en_scores.sort_values(by=["Marker"], inplace=True)

    # Calculate the mean performance for EN
    en_mean = en_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean = en_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean["Marker"] = "Mean"
    en_mean["FE"] = 0
    en_mean["HP"] = 0
    en_mean["Network"] = "EN"
    en_scores = pd.concat([en_scores, en_mean], ignore_index=True)

    # Process Null model for adding Mean row
    null_mean = null_model_scores.groupby(["Model", "Biopsy", "Marker"]).mean(numeric_only=True).reset_index()
    null_mean["Marker"] = "Mean"
    null_mean["FE"] = 0
    null_mean["HP"] = 0
    null_model_scores = pd.concat([null_model_scores, null_mean], ignore_index=True)

    # Combine EN and LGBM scores
    combined_en_lgbm_scores = pd.concat([en_scores, lgbm_scores])

    phenotype_scores = pd.read_csv(Path(phenotype_folder, "patient_metrics.csv"))
    # sort the dataframe by the protein
    phenotype_scores = phenotype_scores.sort_values(by="Protein")

    ari_scores = pd.melt(phenotype_scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["ARI Score"],
                         var_name="ARI", value_name="Score")
    # sort the dataframe by the protein
    ari_scores = ari_scores.sort_values(by="Protein")

    jaccard_scores = pd.melt(phenotype_scores, id_vars=["Biopsy", "Protein"],
                             value_vars=["Jaccard"],
                             var_name="Jaccard", value_name="Score")
    # sort the dataframe by the protein
    jaccard_scores = jaccard_scores.sort_values(by="Protein")

    # Create the figure with a grid specification
    fig = plt.figure(figsize=(17, 17), dpi=150)
    gspec = fig.add_gridspec(8, 4)

    # First bar plot (Null & EN)
    ax1 = fig.add_subplot(gspec[0:2, :])
    ax1.text(-0.05, 1, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax1.set_title('Null & EN MAE', rotation='vertical', x=-0.05, y=0.25, fontsize=12)
    ax1 = create_bar_plot_null_model(data=null_model_scores, metric="MAE", ax=ax1)

    # Second bar plot (EN & LGBM)
    ax2 = fig.add_subplot(gspec[2:4, :])
    ax2.text(-0.05, 1, "b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax2.set_title('EN & LGBM MAE', rotation='vertical', x=-0.05, y=0.25, fontsize=12)
    ax2 = create_bar_plot_en_vs_lgbm(data=combined_en_lgbm_scores, metric="MAE", ax=ax2)

    # Third bar plot (ARI)
    ax31 = fig.add_subplot(gspec[4:6, :2])
    ax31.text(-0.05, 1.1, "c", transform=ax31.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')

    ax31 = plot_ari()

    # Fourth bar plot (Silhouette)
    ax32 = fig.add_subplot(gspec[4:6, 2:])
    ax32.text(-0.05, 1.1, "d", transform=ax32.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')
    ax32 = plot_silhouette()

    ax41 = fig.add_subplot(gspec[6:8, :2])
    ax41.text(-0.05, 1.1, "e", transform=ax41.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')
    ax41 = plot_phenotype_ari(ari_scores)

    ax42 = fig.add_subplot(gspec[6:8, 2:])
    ax42.text(-0.05, 1.1, "f", transform=ax42.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')
    ax42 = plot_phenotype_jaccard(jaccard_scores)

    plt.box(False)

    # Ensure tight layout to avoid overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(Path(image_folder, "fig2.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig2.eps"), dpi=300, bbox_inches='tight', format='eps')

    sys.exit()
