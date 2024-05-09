import argparse
from pathlib import Path
import pandas as pd

biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
save_folder = Path("results", "imputed_data", "ae")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AE Reconstruction data generator')
    parser.add_argument('--radius', "-r", type=int, default=15, choices=[0, 15, 30, 60, 90, 120])
    parser.add_argument("--si", action="store_true", help="Use single imputation data")
    parser.add_argument("-m", "--mode", required=True, choices=["ip", "exp"], default="ip")

    args = parser.parse_args()

    radius: int = args.radius
    single_imputation: bool = args.si
    mode: str = args.mode

    print(f"Using radius {radius} µm")
    print(f"Using single imputation: {single_imputation}")
    print(f"Using mode: {mode}")

    if single_imputation:
        load_path = Path("src", "ae", "single_imputation", mode, "mean")
        save_folder = Path(save_folder, "single")
    else:
        load_path = Path("src", "ae", "multi_imputation", mode, "mean")
        save_folder = Path(save_folder, "multi")

    for biopsy in biopsies:
        patient = "_".join(biopsy.split("_")[:-1])
        print(f"Working on biopsy {biopsy}...")
        print("Loading data...")

        pre_treatment: bool = biopsy.endswith("_1")
        biopsy_load_path = Path(load_path, biopsy, f"{radius}", "experiment_run_0")
        biopsy_save_folder = Path(save_folder, mode, patient, f"{radius}")

        print("Using save folder: ", biopsy_save_folder)
        if not biopsy_save_folder.exists():
            biopsy_save_folder.mkdir(parents=True)

        # load the last five predictions data indicated by 5_predictions to 9_predictions and average them together
        predictions = []
        for i in range(5, 10):
            prediction = pd.read_csv(Path(biopsy_load_path, f"{i}_predictions.csv"))
            predictions.append(prediction)

        assert len(predictions) == 5, "Number of predictions is not 5"

        # average the predictions
        average_predictions = pd.concat(predictions).groupby(level=0).mean()
        # add biopsy type
        average_predictions["Biopsy"] = biopsy
        # pre if biopsy ends with _1 else on
        average_predictions["Treatment"] = "PRE" if biopsy.endswith("_1") else "ON"
        average_predictions["Radius"] = radius

        # save
        if pre_treatment:
            save_path = Path(biopsy_save_folder, "pre_treatment.csv")
            # save average predictions
            average_predictions.to_csv(save_path, index=False)
        else:
            save_path = Path(biopsy_save_folder, "on_treatment.csv")
            # save average predictions
            average_predictions.to_csv(save_path, index=False)
