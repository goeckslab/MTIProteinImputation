import os
from pathlib import Path
import pandas as pd
import argparse
from patient_mapping import patient_mapping
from pycaret.regression import *
from sklearn.preprocessing import MinMaxScaler

base_path = Path("src/tma/lgbm")
SHARED_MARKERS = ['AR', 'Ki67', 'CK14', 'aSMA', 'ER', 'HER2', 'EGFR', 'p21', 'Vimentin',
                  'Ecad', 'CK17', 'pERK', 'PR', 'pRB', 'CK19']


def load_train_data(exclude_patient: str):
    # iterate through data/tma/base
    # load all csv files that do not contain the exclude_patient
    # return a pandas dataframe
    train_data = []
    for root, dirs, files in os.walk("data/tma/base"):
        for name in files:
            file_patient = patient_mapping[name.split(".")[0]]
            if Path(name).suffix == ".csv" and exclude_patient not in file_patient:
                train_data.append(pd.read_csv(os.path.join(root, name)))

    df = pd.concat(train_data, axis=0)
    df = df[SHARED_MARKERS]
    return df


def load_test_data(core: str):
    # load the core
    df = pd.read_csv(f"data/tma/base/{core}.csv")
    df = df[SHARED_MARKERS]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", required=True)
    parser.add_argument("--marker", "-m", help="marker", required=True, choices=SHARED_MARKERS)

    args = parser.parse_args()
    core = args.core
    marker = args.marker

    experiment_id = 0
    base_path = Path(base_path, core, marker)
    save_path = Path(str(base_path), str(experiment_id))
    while Path(save_path).exists():
        experiment_id += 1
        save_path = Path(str(base_path), str(experiment_id))

    save_path.mkdir(parents=True, exist_ok=True)

    # get patient id from core
    patient = patient_mapping[core]

    train_df = load_train_data(exclude_patient=patient)
    test_df = load_test_data(core=args.core)

    # reset index
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # normalize data
    train_df = pd.DataFrame(MinMaxScaler().fit_transform(train_df), columns=SHARED_MARKERS)
    test_df = pd.DataFrame(MinMaxScaler().fit_transform(test_df), columns=SHARED_MARKERS)

    # normalize using min max scaler
    test_df = pd.DataFrame(MinMaxScaler().fit_transform(test_df), columns=SHARED_MARKERS)
    # drop marker from test df
    test_df = test_df.drop(columns=[marker])

    experiment = setup(data=train_df, target=marker, verbose=False, normalize=False)
    regressor = experiment.create_model("lightgbm", verbose=False)
    predictions = experiment.predict_model(regressor, data=test_df, verbose=False)
    results = experiment.pull()

    scores = [{
        "Patient": patient,
        "Biopsy": core,
        "Experiment Id": experiment_id,
        "Model": "LGBM",
        "MAE": results["MAE"].values[0],
        "MSE": results["MSE"].values[0],
        "RMSE": results["RMSE"].values[0],
        "MAPE": results["MAPE"].values[0],
        "R2": results["R2"].values[0]
    }]

    scores = pd.DataFrame(scores)
    scores.to_csv(Path(save_path, f"scores.csv"), index=False)
