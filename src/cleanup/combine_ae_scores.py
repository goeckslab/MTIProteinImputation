import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse, os
import shutil
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', "-m", type=str, required=True, help="The ae model to evaluate",
                        choices=["ae", "ae_m"])
    parser.add_argument('--tma', "--tma", )
    args = parser.parse_args()

    model = args.model
    if model == "ae":
        load_path = Path("src", "ae", "single_imputation")
    elif model == "ae_m":
        load_path = Path("src", "ae", "multi_imputation")
    else:
        raise ValueError("Model not supported")

    all_scores = []
    loaded_files = 0
    for root, sub_directories, files in os.walk(load_path):
        for sub_directory in sub_directories:
            current_path = Path(root, sub_directory)
            try:
                if 'experiment_run' not in str(current_path):  # or int(current_path.stem.split('_')[-1]) > 30:
                    continue
                print(f"Loading path: {Path(root, sub_directory)}")
                scores = pd.read_csv(Path(root, sub_directory, "scores.csv"))
                scores["Load Path"] = str(Path(current_path))

                if "Type" in scores.columns and scores["Type"].iloc[0] in ["ip", "exp"]:
                    print("Correcting type column...")
                    mode = scores["Type"].iloc[0]
                    scores["Mode"] = mode
                if "Mode" in scores.columns and scores["Mode"].iloc[0] in ["GNN", "AE"]:
                    print("Correcting mode column...")
                    network = scores["Mode"].iloc[0]
                    scores["Network"] = network

                if 'Type' in scores.columns:
                    # delete column type
                    del scores["Type"]

                all_scores.append(scores)

                loaded_files += 1

            except:
                continue

    all_scores = pd.concat(all_scores)

    assert ([60, 0, 120, 30, 90, 15] in
            all_scores[all_scores["Mode"] == "exp"]["FE"].unique()), "FE column is not correct"
    assert ([60, 0, 120, 30, 90, 15] in
            all_scores[all_scores["Mode"] == "ip"]["FE"].unique()), "FE column is not correct"

    # print all unique FE for scores which are EXP
    print(all_scores[all_scores["Mode"] == "exp"]["FE"].unique())
    print(all_scores[all_scores["Mode"] == "ip"]["FE"].unique())

    if model == 'ae':
        save_path = Path("results", "temp_scores", "ae", "single_imputation")
    elif model == 'ae_m':
        save_path = Path("results", "temp_scores", "ae", "multi_imputation")
    else:
        raise ValueError("Model not supported")

    if save_path.exists():
        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    all_scores.to_csv(Path(save_path, "scores.csv"), index=False)
    print(f"Loaded {loaded_files} files.")
    print(f"Saved scores to {save_path}")
