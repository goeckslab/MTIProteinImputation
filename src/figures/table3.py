import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    # load ae scores
    ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    microns = [0, 15, 60, 120]
    categories = ["0 µm", "15 µm", "60 µm", "120 µm"]

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

    # rename 23 to 15, 92 to 60 and 184 to 120
    ae_scores["FE"] = ae_scores["FE"].cat.rename_categories(categories)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    # ae_scores = ae_scores[np.abs(ae_scores["MAE"] - ae_scores["MAE"].mean()) <= (3 * ae_scores["MAE"].std())]

    # sort by marker and FE
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # load ae multi scores
    ae_m_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))

    # sort by markers
    # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    ae_m_scores = ae_m_scores[ae_m_scores["FE"].isin(microns)]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_m_scores = ae_m_scores[
        (ae_m_scores["Mode"] == "EXP") & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0) & (
                ae_m_scores["HP"] == 0)]

    # Add µm to the FE column
    ae_m_scores["FE"] = ae_m_scores["FE"].astype(str) + " µm"
    ae_m_scores["FE"] = pd.Categorical(ae_m_scores['FE'], categories)
    # rename 23 to 15, 92 to 60 and 184 to 120
    ae_m_scores["FE"] = ae_m_scores["FE"].cat.rename_categories(categories)
    # sort by marker and FE
    ae_m_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # Remove outliers for MAE and RMSE by only keeping the values that are within +3 to -3 standard deviations
    # ae_m_scores = ae_m_scores[np.abs(ae_m_scores["MAE"] - ae_m_scores["MAE"].mean()) <= (3 * ae_m_scores["MAE"].std())]

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

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)
    # rename MOde EXP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    # get all ap scores
    ap_scores = all_scores[all_scores["Mode"] == "AP"]

    print(ap_scores.groupby(["Network", "FE"])["MAE"].agg(["mean", "std"]))
