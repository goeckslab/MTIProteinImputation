import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

    variance_df = []
    for biopsy in BIOPSIES:
        original_data = pd.read_csv(f"data/bxs/{biopsy}.csv")
        original_data = original_data[SHARED_MARKERS]

        imputed_data = load_imputed_data(biopsy)

        # for every marker calculate variance
        for protein in SHARED_MARKERS:
            original_variance = original_data[protein].var()
            imputed_variance = imputed_data[protein].var()
            # print(
            #    f"Biopsy: {biopsy}, Protein: {protein}, Original variance: {original_variance}, Imputed variance: {imputed_variance}")
            variance_df.append({"Biopsy": biopsy, "Protein": protein, "Original Variance": original_variance,
                                "Imputed Variance": imputed_variance})

    variance_df = pd.DataFrame(variance_df)

    # calculate mean variance for all markers over all biopsies
    mean_variance = variance_df.groupby("Protein").mean()
    mean_variance.reset_index(inplace=True)

    # metl the dataframe into a new dataframe containing only the Original and Imputed variance scores
    mean_variance = pd.melt(mean_variance, id_vars=["Protein"], value_vars=["Original Variance", "Imputed Variance"],
                            var_name="Variance", value_name="Score")
    # sort the dataframe by the protein
    mean_variance = mean_variance.sort_values(by="Protein")

    # melt the dataframe into a new dataframe containing only the Original and Imputed silhouette scores
    silhouette_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                                value_vars=["Original Silhouette Score", "Imputed Silhouette Score"],
                                var_name="Silhouette", value_name="Score")
    # sort the dataframe by the protein
    silhouette_scores = silhouette_scores.sort_values(by="Protein")

    ari_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["ARI Score"],
                         var_name="ARI", value_name="Score")
    # sort the dataframe by the protein
    ari_scores = ari_scores.sort_values(by="Protein")

    #compactness_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
    #                             value_vars=["Original Compactness Score", "Imputed Compactness Score"],
    #                             var_name="Compactness", value_name="Score")
    # sort the dataframe by the protein
    #compactness_scores = compactness_scores.sort_values(by="Protein")

    ami_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["AMI"],
                         var_name="AMI", value_name="Score")
    # sort the dataframe by the protein
    ami_scores = ami_scores.sort_values(by="Protein")

    jaccard_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                             value_vars=["Jaccard"],
                             var_name="Jaccard", value_name="Score")
    # sort the dataframe by the protein
    jaccard_scores = jaccard_scores.sort_values(by="Protein")

    cv_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                        value_vars=["Original CV Score", "Imputed CV Score"],
                        var_name="CV", value_name="Score")
    # sort the dataframe by the protein
    cv_scores = cv_scores.sort_values(by="Protein")

    # plot bar plots of all scores
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    sns.barplot(data=silhouette_scores, x="Protein", y="Score", hue="Silhouette", ax=axs[0, 0])
    axs[0, 0].set_ylabel("Silhouette Score")
    axs[0, 0].set_xlabel("Protein")
    axs[0, 0].set_title("Silhouette Scores for Original and Imputed Data")
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)

    sns.barplot(data=ari_scores, x="Protein", y="Score", hue="ARI", ax=axs[0, 1])
    axs[0, 1].set_ylabel("ARI Score")
    axs[0, 1].set_xlabel("Protein")
    axs[0, 1].set_title("ARI Scores for Original and Imputed Data")
    axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)

    sns.barplot(data=ami_scores, x="Protein", y="Score", hue="AMI", ax=axs[1, 0])
    axs[1, 0].set_ylabel("AMI Score")
    axs[1, 0].set_xlabel("Protein")
    axs[1, 0].set_title("AMI Scores for Original and Imputed Data")
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)

    sns.barplot(data=cv_scores, x="Protein", y="Score", hue="CV", ax=axs[1, 1])
    axs[1, 1].set_ylabel("CV Score")
    axs[1, 1].set_xlabel("Protein")
    axs[1, 1].set_title("CV Scores for Original and Imputed Data")
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45)

    sns.barplot(data=jaccard_scores, x="Protein", y="Score", hue="Jaccard", ax=axs[2, 0])
    axs[2, 0].set_ylabel("Jaccard Score")
    axs[2, 0].set_xlabel("Protein")
    axs[2, 0].set_title("Jaccard Scores for Original and Imputed Data")
    axs[2, 0].set_xticklabels(axs[2, 0].get_xticklabels(), rotation=45)

    sns.barplot(data=mean_variance, x="Protein", y="Score", hue="Variance", ax=axs[2, 1])
    axs[2, 1].set_ylabel("Variance Score")
    axs[2, 1].set_xlabel("Protein")
    axs[2, 1].set_title("Variance Scores for Original and Imputed Data")
    axs[2, 1].set_xticklabels(axs[2, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()



    # plot accuracy in a new plot for each biopsy
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for i, biopsy in enumerate(BIOPSIES):
        data = cv_scores[cv_scores["Biopsy"] == biopsy]
        sns.barplot(data=data, x="Protein", y="Score", hue="CV", ax=axs[i % 2, i // 2], palette={"Original CV Score": "yellow", "Imputed CV Score": "lightgreen"})
        axs[i % 2, i // 2].set_title(f"Biopsy: {' '.join(biopsy.split('_'))}")
        axs[i % 2, i // 2].set_xticklabels(axs[i % 2, i // 2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()