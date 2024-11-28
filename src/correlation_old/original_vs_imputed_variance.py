from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    variance_df = []
    for biopsy in BIOPSIES:
        original_data = pd.read_csv(f"data/bxs/{biopsy}.csv")
        original_data = original_data[SHARED_MARKERS]

        imputed_data = load_imputed_data(biopsy)

        # for every marker calculate variance
        for marker in SHARED_MARKERS:
            original_variance = original_data[marker].var()
            imputed_variance = imputed_data[marker].var()
            print(
                f"Biopsy: {biopsy}, Marker: {marker}, Original variance: {original_variance}, Imputed variance: {imputed_variance}")
            variance_df.append({"Biopsy": biopsy, "Marker": marker, "Original Variance": original_variance,
                                "Imputed Variance": imputed_variance})

    variance_df = pd.DataFrame(variance_df)
    print(variance_df)

    # calculate mean variance for all markers over all biopsies
    mean_variance = variance_df.groupby("Marker").mean()
    mean_variance.reset_index(inplace=True)

    print(mean_variance)

    # metl the dataframe into a new dataframe containing only the Original and Imputed variance scores
    mean_variance = pd.melt(mean_variance, id_vars=["Marker"], value_vars=["Original Variance", "Imputed Variance"],
                            var_name="Variance", value_name="Score")

    # plot the variance over all biopsies between original and imputed data
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    sns.barplot(data=mean_variance, x="Marker", y="Score", hue="Variance", ax=axs)
    axs.set_ylabel("Variance")
    axs.set_xlabel("Marker")
    axs.set_title("Variance for Original and Imputed Data")
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45)
    plt.show()
    plt.close()
