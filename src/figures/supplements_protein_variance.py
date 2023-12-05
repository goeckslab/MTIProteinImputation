import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd
import os
import numpy as np

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    base_path: Path = Path("results", "supplements", "protein_variance")

    if not base_path.exists():
        base_path.mkdir(parents=True)

    variance_scores = []
    for patient in PATIENTS:
        save_path = Path(base_path, patient)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        patient_data = []
        # iterate through the data bxs folder
        for root, dirs, files in os.walk(Path("data", "bxs")):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                # if file is in subfolder ignore
                if Path(root).name != "bxs":
                    continue

                if patient in file:
                    patient_data.append(pd.read_csv(Path(root, file)))

        patient_data = pd.concat(patient_data)

        for protein in patient_data.columns:
            # calculate variance for each protein
            variance = patient_data[protein].var()
            patient = " ".join(patient.split("_")[:2])
            variance_scores.append({"Protein": protein, "Variance": variance, "Patient": patient})

    variance_scores = pd.DataFrame(variance_scores)
    # sort variance scores descending
    variance_scores.sort_values(by=["Variance"], ascending=False, inplace=True)

    # select only shared proteins
    variance_scores = variance_scores[variance_scores["Protein"].isin(SHARED_PROTEINS)]
    # reindex
    variance_scores.reset_index(drop=True, inplace=True)
    # sort by protein and variance descending
    variance_scores.sort_values(by=["Protein", "Variance"], ascending=False, inplace=True)
    # save variance scores
    variance_scores.to_csv(Path(base_path, "protein_variance_scores.csv"), index=False)

    # reset index and keep index
    variance_scores.reset_index(inplace=True)
    # calculate the mean index. Group by protein and calculate the mean of the index column
    mean_index = variance_scores.groupby(["Protein"])["index"].mean().reset_index()
    # calculate mean variance
    mean_variance = variance_scores.groupby(["Protein"])["Variance"].mean().reset_index()

    # merge mean index and mean variance
    mean_index = mean_index.merge(mean_variance, on=["Protein"])

    mean_index.reset_index(inplace=True, drop=True)
    mean_index.reset_index(inplace=True)
    # rename index to mean index position and rename level_0 to rank
    mean_index.rename(columns={"index": "Mean Index Position", "level_0": "Rank"}, inplace=True)

    # remove Mean Index Position
    mean_index.drop(columns=["Mean Index Position", "Rank"], inplace=True)
    # sort ascending
    mean_index.sort_values(by=["Variance"], ascending=False, inplace=True)

    mean_index.to_csv(Path(base_path, "T3.csv"), index=False)
