import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pathlib import Path
import numpy as np

save_path = Path("results", "supplements", "mean_performance")

def load_ae_scores(single_imputation: bool, in_patient: bool) -> pd.DataFrame:
    if single_imputation:
        ae_scores = pd.read_csv(Path("results", "scores", "ae", "scores.csv"))
    else:
        ae_scores = pd.read_csv(Path("results", "scores", "ae_m", "scores.csv"))

        # sort by markers
        # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    ae_scores = ae_scores[ae_scores["FE"].isin(microns)]

    # select only EXP mode, mean replace value, no noise and no hp in a one line statement
    ae_scores = ae_scores[
        (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0) & (
                ae_scores["HP"] == 0)]

    if in_patient:
        ae_scores = ae_scores[ae_scores["Mode"] == "IP"]
    else:
        ae_scores = ae_scores[ae_scores["Mode"] == "EXP"]

    # Add µm to the FE column
    ae_scores["FE"] = ae_scores["FE"].astype(str) + " µm"
    ae_scores["FE"] = pd.Categorical(ae_scores['FE'], categories)

    # rename categories
    ae_scores["FE"] = ae_scores["FE"].cat.rename_categories(categories)

    # sort by marker and FE
    ae_scores.sort_values(by=["Marker", "FE"], inplace=True)
    return ae_scores


def load_lgbm_scores(in_patient: bool) -> pd.DataFrame:
    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))

    # select only the scores for the 0 µm, 15 µm, 60 µm, 120 µm
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin(microns)]
    # select exp scores
    if in_patient:
        lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "IP"]
    else:
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
    if not save_path.exists():
        save_path.mkdir(parents=True)


    microns = [0, 15, 30, 60, 90, 120]
    categories = ["0 µm", "15 µm", "30 µm", "60 µm", "90 µm", "120 µm"]

    ae_scores = load_ae_scores(True, False)
    ae_m_scores = load_ae_scores(False, False)
    lgbm_scores = load_lgbm_scores(False)

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)
    # rename MOde EXP to AP
    all_scores["Mode"] = all_scores["Mode"].replace({"EXP": "AP"})

    # get all ap scores
    ap_scores = all_scores[all_scores["Mode"] == "AP"]
    # assert only AP mode in ap scores
    assert len(ap_scores["Mode"].unique()) == 1
    print("AP Mean & Std")
    print(ap_scores.groupby(["Network", "FE"])["MAE"].agg(["mean", "std"]))
    # save ap scores
    ap_scores.to_csv(Path(save_path, "T2.csv"), index=False)


    ae_scores = load_ae_scores(True, True)
    ae_m_scores = load_ae_scores(False, True)
    lgbm_scores = load_lgbm_scores(True)

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, ae_scores, ae_m_scores], axis=0)

    # remove column hyper, experiment, Noise, Replace Value
    all_scores.drop(columns=["HP", "Experiment", "Noise", "Replace Value"], inplace=True)

    # get all ap scores
    ip_scores = all_scores[all_scores["Mode"] == "IP"]
    # assert only AP mode in ap scores
    assert len(ip_scores["Mode"].unique()) == 1
    print("IP Mean & Std")
    print(ip_scores.groupby(["Network", "FE"])["MAE"].agg(["mean", "std"]))

    # save ip scores
    ip_scores.to_csv(Path(save_path, "T1.csv"), index=False)

