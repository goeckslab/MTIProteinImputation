import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

load_folder = Path("results", "phenotypes")
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']
PROTEINS_OF_INTEREST = ["aSMA", "CD45", "CK19", "CK14", "CK17"]
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
COLOR_PALETTE = {
    "CD45": (0.7107843137254903, 0.7843137254901962, 0.8813725490196078, 1.0),
    "CK19": (0.8818627450980391, 0.5053921568627451, 0.17303921568627467, 1.0),
    "aSMA": (0.22941176470588232, 0.5705882352941177, 0.22941176470588232, 1.0),
    "CK14": (0.948529411764706, 0.6455882352941177, 0.6397058823529412, 1.0),
    "CK17": (0.5171568627450981, 0.3583333333333334, 0.3259803921568628, 1.0)
}


def load_imputed_data(biopsy: str):
    patient = '_'.join(biopsy.split("_")[0:2])
    pre: bool = True if "1" == biopsy.split("_")[-1] else False
    if pre:
        imp_treatment = pd.read_csv(
            Path("results", "imputed_data", "ae", "single", "exp", patient, "0", "pre_treatment.csv"))
    else:
        imp_treatment = pd.read_csv(
            Path("results", "imputed_data", "ae", "single", "exp", patient, "0", "on_treatment.csv"))

    return imp_treatment


if __name__ == '__main__':
    scores = pd.read_csv(Path(load_folder, "patient_metrics.csv"))

    # melt the dataframe into a new dataframe containing only the Original and Imputed silhouette scores
    silhouette_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                                value_vars=["Original Silhouette Score", "Imputed Silhouette Score"],
                                var_name="Silhouette", value_name="Score")
    # sort the dataframe by the protein
    silhouette_scores = silhouette_scores.sort_values(by="Protein")
    silhouette_scores = pd.concat([silhouette_scores] * 30, ignore_index=True)

    ami_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["AMI"],
                         var_name="AMI", value_name="Score")
    # sort the dataframe by the protein
    ami_scores = ami_scores.sort_values(by="Protein")

    # plot bar plots of all scores
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    sns.barplot(data=silhouette_scores, x="Protein", y="Score", hue="Silhouette", ax=axs[0])
    axs[0].set_ylabel("Silhouette Score")
    axs[0].set_xlabel("Protein")
    axs[0].set_title("Silhouette Scores for Original and Imputed Data")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)

    # change legend handles
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles=handles[0:], labels=["Original", "Imputed"], loc="upper center", bbox_to_anchor=(0.5, 0.92),
                  ncol=2)

    # remove boxes
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)

    # Add statistical annotations
    pairs = [
        (("CD45", "Original Silhouette Score"), ("CD45", "Imputed Silhouette Score")),
        (("CK19", "Original Silhouette Score"), ("CK19", "Imputed Silhouette Score")),
        (("aSMA", "Original Silhouette Score"), ("aSMA", "Imputed Silhouette Score")),
        (("CK14", "Original Silhouette Score"), ("CK14", "Imputed Silhouette Score")),
        (("CK17", "Original Silhouette Score"), ("CK17", "Imputed Silhouette Score")),
    ]

    order = PROTEINS_OF_INTEREST
    hue_order = ["Original Silhouette Score", "Imputed Silhouette Score"]
    annotator = Annotator(axs[0], pairs, data=silhouette_scores, x="Protein", y="Score", order=order, hue="Silhouette",
                          hue_order=hue_order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',
                        comparisons_correction="Benjamini-Hochberg")

    annotator.apply_and_annotate()

    sns.barplot(data=ami_scores, x="Protein", y="Score", ax=axs[1], palette=COLOR_PALETTE)
    axs[1].set_ylabel("AMI Score")
    axs[1].set_xlabel("Protein")
    axs[1].set_title("AMI Scores for Original and Imputed Data")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)
    axs[1].set_ylim(0, 1)

    # remove boxes
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path("figures", "supplements", "phenotype_scores.png"), dpi=150)
