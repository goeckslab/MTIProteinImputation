import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pathlib import Path
import numpy as np


def load_ae_scores(single_imputation: bool) -> pd.DataFrame:
    if single_imputation:
        ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    else:
        ae_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))

        # sort by markers
        # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    ae_scores = ae_scores[ae_scores["FE"].isin(microns)]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_scores = ae_scores[
        (ae_scores["Mode"] == "EXP") & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0) & (
                ae_scores["HP"] == 0)]

    # Add µm to the FE column
    ae_scores["FE"] = ae_scores["FE"].astype(str) + " µm"
    ae_scores["FE"] = pd.Categorical(ae_scores['FE'], categories)

    # rename categories
    ae_scores["FE"] = ae_scores["FE"].cat.rename_categories(categories)

    # sort by marker and FE
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)
    return ae_scores


def load_lgbm_scores() -> pd.DataFrame:
    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))

    # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin(microns)]
    # select exp scores
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]
    # only select non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]

    # Add µm to the FE column
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], categories)

    # rename 23 to 15, 92 to 60 and 184 to 120
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(categories)

    # sort by marker and FE
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    return lgbm_scores


if __name__ == '__main__':
    microns = [0, 15, 60, 120]
    categories = ["0 µm", "15 µm", "60 µm", "120 µm"]

    ae_scores = load_ae_scores(True)
    ae_m_scores = load_ae_scores(False)
    lgbm_scores = load_lgbm_scores()

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)
    # rename MOde EXP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    # get all ap scores
    ap_scores = all_scores[all_scores["Mode"] == "AP"]

    print(len(ap_scores))

    print(ap_scores.groupby(["Network", "FE"])["MAE"].agg(["mean", "std"]))
