from pathlib import Path
import os
import json
import pandas as pd

load_path = Path("src/tma/ae")
save_folder = Path("results/tma")

if __name__ == '__main__':

    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    scores = []
    # iterate through the load path
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if name == "scores.csv":
                data = pd.read_csv(os.path.join(root, name))

                # use iteration 5-10 to calculate mean performance
                data = data[(data["Iteration"] >= 5) & (data["Iteration"] <= 10)]
                data = data.groupby(["Biopsy", "Patient", "Marker", "Experiment"]).mean(numeric_only=True)
                data = data.reset_index()
                data["Model"] = "AE"
                # rename Experiment to Epxierment Id
                data = data.rename(columns={"Experiment": "Experiment Id"})
                scores.append(data)
    scores = pd.concat(scores)
    scores.to_csv(Path(save_folder, "ae_scores.csv"), index=False)
