import argparse
from pathlib import Path
import pandas as pd

biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
save_folder = Path("results", "imputed_data", "en")
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':

    load_path = Path("src", "en", "exp_patient")
    save_folder = Path(save_folder)

    for biopsy in biopsies:
        patient = "_".join(biopsy.split("_")[:-1])
        print(f"Working on biopsy {biopsy}...")
        print("Loading data...")

        pre_treatment: bool = biopsy.endswith("_1")

        biopsy_save_folder = Path(save_folder, "exp", patient)

        print("Using save folder: ", biopsy_save_folder)
        if not biopsy_save_folder.exists():
            biopsy_save_folder.mkdir(parents=True)

        predictions = pd.DataFrame(columns=SHARED_MARKERS)
        # iterate through SHARED MARKERS
        for marker in SHARED_MARKERS:
            biopsy_load_path = Path(load_path, biopsy, biopsy, marker, "experiment_run_0")
            # load predictions.csv
            prediction: pd.DataFrame = pd.read_csv(Path(biopsy_load_path, f"{marker}_predictions.csv"), header=None)
            # add prediction to predictions df
            predictions[marker] = prediction[0].values

        predictions["Biopsy"] = biopsy
        # pre if biopsy ends with _1 else on
        predictions["Treatment"] = "PRE" if biopsy.endswith("_1") else "ON"
        predictions["Radius"] = 0

        # save predictions
        if pre_treatment:
            save_path = Path(biopsy_save_folder, "pre_treatment.csv")
            # save average predictions
            predictions.to_csv(save_path, index=False)
        else:
            save_path = Path(biopsy_save_folder, "on_treatment.csv")
            # save average predictions
            predictions.to_csv(save_path, index=False)
