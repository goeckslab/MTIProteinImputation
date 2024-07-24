import os, shutil, random
import numpy as np
from sklearn.linear_model import ElasticNetCV
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, \
    root_mean_squared_error
from patient_mapping import patient_mapping
from sklearn.preprocessing import MinMaxScaler

base_path = Path("src/tma/en")
l1_ratios = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def load_train_data(exclude_patient: str):
    # iterate through data/tma/base
    # load all csv files that do not contain the exclude_patient
    # return a pandas dataframe
    train_data = []
    for root, dirs, files in os.walk("data/tma/base"):
        for name in files:
            file_patient = patient_mapping[name.split(".")[0]]
            if Path(name).suffix == ".csv" and exclude_patient not in file_patient:
                # print(f"Loading {name}")
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
    # print(f"Excluding patient: {patient}")

    train_df = load_train_data(exclude_patient=patient)
    test_df = load_test_data(core=args.core)

    # use sklearn min max scaler to scale the data
    train_df = pd.DataFrame(MinMaxScaler().fit_transform(train_df), columns=SHARED_MARKERS)
    test_df = pd.DataFrame(MinMaxScaler().fit_transform(test_df), columns=SHARED_MARKERS)

    X = train_df.drop(columns=[args.marker])
    y = train_df[args.marker]

    # assert that marker is not in X
    assert args.marker not in X.columns, f"{args.marker} is in the dataset"

    # generate random number between 0 and 1
    l1_ratio = random.choice(l1_ratios)
    elastic_net = ElasticNetCV(cv=5, random_state=random.randint(0, 10000), l1_ratio=l1_ratio)
    elastic_net.fit(X, y)

    X_test = test_df.drop(columns=[args.marker])
    y_test = test_df[args.marker]

    # assert that marker is not in X_test
    assert args.marker not in X_test.columns, f"{args.marker} is in the dataset"

    y_hat = elastic_net.predict(X_test)
    y_hat_df = pd.DataFrame(y_hat, columns=[args.marker])

    y_hat_df.to_csv(Path(save_path, f"{args.marker}_predictions.csv"), index=False, header=False)

    data = {
        "biopsy": core,
        "patient": patient,
        "mode": "EXP",
        "mean_squared_error": mean_squared_error(y_test, y_hat),
        "mean_absolute_error": mean_absolute_error(y_test, y_hat),
        "root_mean_squared_error": root_mean_squared_error(y_test, y_hat),
        "mape": mean_absolute_percentage_error(y_test, y_hat),
        "marker": args.marker,
        "model": "elastic_net",
        "l1_ratio": l1_ratio,
        "experiment_id": experiment_id
    }

    with open(Path(save_path, 'evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
