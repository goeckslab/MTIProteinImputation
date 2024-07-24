from pathlib import Path
import os
import json
import pandas as pd

load_path = Path("src/tma/lgbm")
save_folder = Path("results/tma")

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':

    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    scores = []
    # iterate through the load path
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if name == "scores.csv":
                data = pd.read_csv(Path(root, name))
                marker = str(Path(root, name)).split("/")[-3]
                assert marker in SHARED_MARKERS, f"Marker {marker} not in SHARED_MARKERS"

                scores.append({
                    "Biopsy": data["Biopsy"].values[0],
                    "Patient": data["Patient"].values[0],
                    "Marker": marker,
                    "MSE": data["MSE"].values[0],
                    "MAPE": data["MAPE"].values[0],
                    "MAE": data["MAE"].values[0],
                    "RMSE": data["RMSE"].values[0],
                    "Model": "LGBM",
                    "Experiment Id": data["Experiment Id"].values[0]
                })

    scores = pd.DataFrame(scores)

    scores.to_csv(Path(save_folder, "lgbm_scores.csv"), index=False)
