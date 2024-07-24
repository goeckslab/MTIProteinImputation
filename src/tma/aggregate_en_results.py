from pathlib import Path
import os
import json
import pandas as pd

load_path = Path("src/tma/en")
save_folder = Path("results/tma")

if __name__ == '__main__':

    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    scores = []
    # iterate through the load path
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if name == "evaluation.json":
                data = json.load(open(os.path.join(root, name)))

                scores.append({
                    "Biopsy": data["biopsy"],
                    "Patient": data["patient"],
                    "Marker": data["marker"],
                    "MSE": data["mean_squared_error"],
                    "MAPE": data["mape"],
                    "MAE": data["mean_absolute_error"],
                    "RMSE": data["root_mean_squared_error"],
                    "Model": "EN",
                    "Experiment Id": data["experiment_id"]
                })

    scores = pd.DataFrame(scores)

    scores.to_csv(Path(save_folder, "en_scores.csv"), index=False)
