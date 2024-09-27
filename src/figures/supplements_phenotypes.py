import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

load_folder = Path("results", "phenotypes")
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]


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
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    sns.barplot(data=silhouette_scores, x="Protein", y="Score", hue="Silhouette", ax=axs[0])
    axs[0].set_ylabel("Silhouette Score")
    axs[0].set_xlabel("Protein")
    axs[0].set_title("Silhouette Scores for Original and Imputed Data")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)

    # Add statistical annotations
    pairs = [
        (("pRB", "Original Silhouette Score"), ("pRB", "Imputed Silhouette Score")),
        (("CD45", "Original Silhouette Score"), ("CD45", "Imputed Silhouette Score")),
        (("CK19", "Original Silhouette Score"), ("CK19", "Imputed Silhouette Score")),
        (("Ki67", "Original Silhouette Score"), ("Ki67", "Imputed Silhouette Score")),
        (("aSMA", "Original Silhouette Score"), ("aSMA", "Imputed Silhouette Score")),
        (("Ecad", "Original Silhouette Score"), ("Ecad", "Imputed Silhouette Score")),
        (("PR", "Original Silhouette Score"), ("PR", "Imputed Silhouette Score")),
        (("CK14", "Original Silhouette Score"), ("CK14", "Imputed Silhouette Score")),
        (("HER2", "Original Silhouette Score"), ("HER2", "Imputed Silhouette Score")),
        (("AR", "Original Silhouette Score"), ("AR", "Imputed Silhouette Score")),
        (("CK17", "Original Silhouette Score"), ("CK17", "Imputed Silhouette Score")),
        (("p21", "Original Silhouette Score"), ("p21", "Imputed Silhouette Score")),
        (("Vimentin", "Original Silhouette Score"), ("Vimentin", "Imputed Silhouette Score")),
        (("pERK", "Original Silhouette Score"), ("pERK", "Imputed Silhouette Score")),
        (("EGFR", "Original Silhouette Score"), ("EGFR", "Imputed Silhouette Score")),
        (("ER", "Original Silhouette Score"), ("ER", "Imputed Silhouette Score")),
    ]

    order = SHARED_MARKERS
    hue_order = ["Original Silhouette Score", "Imputed Silhouette Score"]
    annotator = Annotator(axs[0], pairs, data=silhouette_scores, x="Protein", y="Score", order=order, hue="Silhouette",
                          hue_order=hue_order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")

    annotator.apply_and_annotate()

    sns.barplot(data=ami_scores, x="Protein", y="Score", hue="AMI", ax=axs[1])
    axs[1].set_ylabel("AMI Score")
    axs[1].set_xlabel("Protein")
    axs[1].set_title("AMI Scores for Original and Imputed Data")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(Path("figures", "supplements", "phenotype_scores.png"), dpi=150)
