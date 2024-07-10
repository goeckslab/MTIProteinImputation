import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
from sklearn.linear_model import ElasticNetCV

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
save_path = Path("results", "scores", "null_model")


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


def load_train_data(base_path: Path, patient: str):
    train_data = []
    for file in os.listdir(base_path):
        file_name = Path(file).stem
        if file.endswith(".csv") and patient not in file_name:
            print("Loading train file: " + file)
            data = pd.read_csv(Path(base_path, file))
            data = clean_column_names(data)
            train_data.append(data)

    assert len(train_data) == 6, f"There should be 6 train datasets, loaded {len(train_data)}"
    return pd.concat(train_data)


if __name__ == '__main__':

    if not save_path.exists():
        save_path.mkdir(parents=True)

    scores = []
    for biopsy in BIOPSIES:
        print("Working on biopsy: ", biopsy)
        patient = "_".join(biopsy.split("_")[:-1])
        # load biopsy data
        test_data = pd.read_csv(Path("data", "bxs", f"{biopsy}.csv"))
        train_data = load_train_data(Path("data", "bxs"), patient=patient)

        for protein in SHARED_MARKERS:
            for i in tqdm(range(30)):
                temp_train_data = train_data[SHARED_MARKERS].copy()
                temp_train_data = temp_train_data.drop(protein, axis=1)
                temp_test_data = test_data[SHARED_MARKERS].copy()
                temp_test_data = temp_test_data.drop(protein, axis=1)

                ground_truth = test_data[protein]
                y_hat = train_data[protein].sample(frac=0.8).mean()
                y_hat = np.repeat(y_hat, len(ground_truth))

                # run elastic net using ElasticNetCv from sklearn
                model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, n_jobs=-1, max_iter=10000)

                model.fit(temp_train_data, train_data[protein])
                en_y_hat = model.predict(temp_test_data)

                en_mae = mean_absolute_error(en_y_hat, ground_truth)
                en_rmse = root_mean_squared_error(en_y_hat, ground_truth)

                # calculate mae, rmse
                mae = mean_absolute_error(y_hat, ground_truth)
                rmse = root_mean_squared_error(y_hat, ground_truth)

                scores.append(
                    {"Biopsy": biopsy, "Protein": protein, "MAE": en_mae, "RMSE": en_rmse, "Iteration": i,
                     "Model": "EN"})
                scores.append(
                    {"Biopsy": biopsy, "Protein": protein, "MAE": mae, "RMSE": rmse, "Iteration": i, "Model": "Null"})

    scores = pd.DataFrame(scores)

    # sort scores by protein
    scores = scores.sort_values(by=["Protein", "Biopsy"])
    scores.to_csv(Path(save_path, "scores.csv"), index=False)
