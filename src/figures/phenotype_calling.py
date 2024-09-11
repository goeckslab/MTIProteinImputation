import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

load_folder = Path("results", "phenotypes")

if __name__ == '__main__':
    scores = pd.read_csv(Path(load_folder, "silhouette.csv"))

    # remove Biopsy 9_14_2
    scores = scores[scores["Biopsy"] != "9_15_1"]

    # melt the dataframe into a new dataframe containing only the Original and Imputed silhouette scores
    silhouette_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                                value_vars=["Original Silhouette Score", "Imputed Silhouette Score"],
                                var_name="Silhouette", value_name="Score")

    ari_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["ARI Score"],
                         var_name="ARI", value_name="Score")

    print(ari_scores)

    compactness_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                                 value_vars=["Original Compactness Score", "Imputed Compactness Score"],
                                 var_name="Compactness", value_name="Score")

    cv_scores = pd.melt(scores, id_vars=["Biopsy", "Protein"],
                        value_vars=["Original CV Score", "Imputed CV Score"],
                        var_name="CV", value_name="Score")

    # plot bar plots of all scores
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
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

    sns.barplot(data=compactness_scores, x="Protein", y="Score", hue="Compactness", ax=axs[1, 0])
    axs[1, 0].set_ylabel("Compactness Score")
    axs[1, 0].set_xlabel("Protein")
    axs[1, 0].set_title("Compactness Scores for Original and Imputed Data")
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)

    sns.barplot(data=cv_scores, x="Protein", y="Score", hue="CV", ax=axs[1, 1])
    axs[1, 1].set_ylabel("CV Score")
    axs[1, 1].set_xlabel("Protein")
    axs[1, 1].set_title("CV Scores for Original and Imputed Data")
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()
