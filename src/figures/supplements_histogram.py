import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']
save_folder = Path("figures", "supplements", "variance")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load biopsy data for patients
    bx_data = {}
    for patient in PATIENTS:
        patient_scores = pd.read_csv(
            Path("data", "bxs", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"), sep="\t")
        patient_scores = patient_scores.loc[(patient_scores != 0.0).any(axis=1)]
        bx_data[patient] = patient_scores

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    # Histograms for each marker
    ax11 = fig.add_subplot(411)
    hist = sns.histplot(bx_data["9_2"]["CK19"], color="blue", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_3"]["CK19"], color="green", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["CK19"], color="yellow", ax=ax11, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["CK19"], color="red", ax=ax11, kde=True, stat="count")
    ax11.set_ylabel("CK19")

    ax12 = fig.add_subplot(412)
    hist = sns.histplot(bx_data["9_2"]["ER"], color="blue", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_3"]["ER"], color="green", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["ER"], color="yellow", ax=ax12, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["ER"], color="red", ax=ax12, kde=True, stat="count")
    ax12.set_ylabel("ER")

    ax13 = fig.add_subplot(413)
    hist = sns.histplot(bx_data["9_2"]["pRB"], color="blue", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_3"]["pRB"], color="green", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["pRB"], color="yellow", ax=ax13, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["pRB"], color="red", ax=ax13, kde=True, stat="count")
    ax13.set_ylabel("pRB")

    ax14 = fig.add_subplot(414)
    hist = sns.histplot(bx_data["9_2"]["CK17"], color="blue", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_3"]["CK17"], color="green", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_14"]["CK17"], color="yellow", ax=ax14, kde=True, stat="count")
    sns.histplot(bx_data["9_15"]["CK17"], color="red", ax=ax14, kde=True, stat="count")
    ax14.set_ylabel("CK17")

    # add legend using Biopsies 9_2, 9_3, 9_14 and 9_15, with colors blue, green, yellow and red respectively
    ax11.legend(["9 2", "9 3", "9 14", "9 15"], title="Biopsy", loc="upper center", bbox_to_anchor=(0.5, 1.4),
                ncol=4)

    plt.tight_layout()
    plt.savefig(Path(save_folder, "variance_histogram.png"), dpi=150)
