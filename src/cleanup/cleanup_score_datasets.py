import pandas as pd
from pathlib import Path
import os, shutil
from tqdm import tqdm
from argparse import ArgumentParser

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def load_lgbm_scores(load_path: Path, mode: str, network: str) -> pd.DataFrame:
    try:
        scores = []
        for root, dirs, files in os.walk(load_path):
            for name in files:
                if Path(name).suffix == ".csv":
                    file_name = os.path.join(root, name)
                    score = pd.read_csv(file_name, sep=",", header=0)
                    if 'Unnamed: 0' in score.columns:
                        score = score.drop(columns=['Unnamed: 0'])

                    scores.append(score)

        assert len(scores) == 8, f"Not all biopsies could be loaded for load path {load_path}"
        scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
        scores["Mode"] = mode
        scores["Network"] = network
        return scores

    except Exception as e:
        print(e)


def prepare_lbgm_scores(save_path: Path):
    try:
        print("Preparing light gbm scores...")
        lgbm_path = Path(save_path, "lgbm")

        if lgbm_path.exists():
            shutil.rmtree(lgbm_path)
        lgbm_path.mkdir(parents=True, exist_ok=True)

        microns = [0, 15, 30, 60, 90, 120]
        scores = []
        for micron in tqdm(microns):
            ip_path: Path = Path("results", "temp_scores", "ip", str(micron))
            exp_path: Path = Path("results", "temp_scores", "exp", str(micron))
            scores.append(load_lgbm_scores(ip_path, "IP", "LGBM"))
            scores.append(load_lgbm_scores(exp_path, "EXP", "LGBM"))

        scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])

        # replace _ with '' for biopsy column
        scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

        # convert Hyper False to 0
        # check if hyper column is type int
        if scores["Hyper"].dtype != "int64":
            scores["Hyper"] = scores["Hyper"].apply(lambda x: 0 if x == "False" else 1)
        # convert Hyper column to int
        scores["Hyper"] = scores["Hyper"].apply(lambda x: int(x))

        # Remove Load Path Random Seed,
        scores = scores.drop(columns=["Load Path", "Random Seed"])
        scores["Network"] = "LGBM"

        if "Hyper" in scores.columns:
            # rename hyper column to hp
            scores = scores.rename(columns={"Hyper": "HP"})

        scores.to_csv(Path(lgbm_path, "scores.csv"), index=False)

    except BaseException as ex:
        print(ex)
        print("LGBM scores could not be cleaned up")
        print(scores)


def prepare_ae_scores(save_path: Path, single: bool):
    print(f"Preparing ae scores")

    if single:
        scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
        network = "AE"
        ae_path = Path(save_path, "ae")
    else:
        scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))
        network = "AE M"
        ae_path = Path(save_path, "ae_m")

    if ae_path.exists():
        shutil.rmtree(ae_path)
    ae_path.mkdir(parents=True, exist_ok=True)

    scores["Mode"] = scores["Mode"].apply(lambda x: x.upper())
    # convert FE column to int
    scores["FE"] = scores["FE"].apply(lambda x: int(x))
    # replace _ with '' for biopsy column
    scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

    # group by marker, biopsy and experiment, only keep iteration 5-9
    scores = scores.groupby(["Marker", "Biopsy", "Experiment", "Mode", "HP", "FE", "Noise", "Replace Value"]).nth(
        [5, 6, 7, 8, 9]).reset_index()

    # calculate mean of MAE scores
    scores = scores.groupby(["Marker", "Biopsy", "Experiment", "Mode", "HP", "FE", "Noise", "Replace Value"]).mean(
        numeric_only=True).reset_index()

    # select only scores of shared markers
    scores = scores[scores["Marker"].isin(SHARED_MARKERS)]

    # remove load path and random seed
    if "Load Path" in scores.columns:
        scores = scores.drop(columns=["Load Path"])

    if imputation == "multi":
        # remove round column:
        scores = scores.drop(columns=["Round"])

    # drop imputation & iteration columns
    scores = scores.drop(columns=["Imputation", "Iteration"])
    if "Network" not in scores.columns:
        scores["Network"] = network
    scores.to_csv(Path(ae_path, "scores.csv"), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, required=True,
                        choices=["lgbm", "ae", "ae_m"])
    args = parser.parse_args()

    # create new scores folder
    save_path = Path("results/scores")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    model = args.model

    if model == "lgbm":
        try:
            prepare_lbgm_scores(save_path=save_path)
        except:
            print("Could not prepare lgbm scores")

    elif model == "ae":
        try:
            prepare_ae_scores(save_path=save_path)
        except:
            print("Could not prepare ae scores")

    elif model == "ae_m":
        try:
            prepare_ae_scores(save_path=save_path, imputation="multi")
        except:
            print("Could not prepare ae multi scores")

    else:
        print("Model not found")
