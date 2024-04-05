import os, argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "pre_post")


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
            data["Treatment"] = "PRE" if "_1" in file else "ON"

            assert "Treatment" in data.columns, f"Treatment column is missing for dataframe of patient {file}"
            train_data.append(data)

    assert len(train_data) == 6, f"There should be 6 train datasets, loaded {len(train_data)}"
    return pd.concat(train_data)


def load_test_data(patient: str) -> pd.DataFrame:
    pre = pd.read_csv(Path("data", "bxs", f"{patient}_1.csv"))
    pre["Treatment"] = "PRE"
    post = pd.read_csv(Path("data", "bxs", f"{patient}_2.csv"))
    post["Treatment"] = "ON"
    return pd.concat([pre, post])


if __name__ == '__main__':

    if not save_path.exists():
        save_path.mkdir(parents=True)

    parser = argparse.ArgumentParser(description='Run pre-post')
    parser.add_argument("-p", "--patient", required=True)
    args = parser.parse_args()

    patient: str = args.patient
    print(f"Working on patient: {patient}")

    # load patient data
    test_data = load_test_data(patient=patient)
    test_data = clean_column_names(test_data)
    train_data = load_train_data(Path("data", "bxs"), patient=patient)
    train_data = clean_column_names(train_data)

    # scale data
    scaler = MinMaxScaler()
    train_treatment = train_data["Treatment"]
    train_data[SHARED_MARKERS] = scaler.fit_transform(train_data[SHARED_MARKERS])
    test_treatment = test_data["Treatment"]
    test_data[SHARED_MARKERS] = scaler.fit_transform(test_data[SHARED_MARKERS])

    scores = []
    for treatment in train_treatment.unique():
        print(f"Working on treatment: {treatment}")
        for target_protein in SHARED_MARKERS:
            print(f"Working on protein: {target_protein}")

            # select only treatment class for the train set
            train_df = train_data[train_treatment == treatment]
            # check that treatment column matches treatment
            assert train_df["Treatment"].unique() == treatment, "Treatment column does not match treatment"

            test_df = test_data[test_treatment != treatment]
            # check that treatment column matches treatment
            assert test_df["Treatment"].unique() != treatment, "Treatment column does not match treatment"
            test_df_treatment = test_df["Treatment"]
            test_df = test_df[SHARED_MARKERS]
            # encode treatment column
            test_df_treatment = test_df_treatment.map({"PRE": 0, "ON": 1})

            for _ in range(30):
                # Creating a bootstrap sample of the training data
                train_df_bootstrap = resample(train_df, replace=True, n_samples=len(train_df))
                y_df_bootstrap = train_df_bootstrap["Treatment"]
                # encode treatment column
                y_df_bootstrap = y_df_bootstrap.map({"PRE": 0, "ON": 1})

                train_df_bootstrap = train_df_bootstrap[SHARED_MARKERS]

                layers = 128
                # create non linear model
                model = Sequential(
                    [Dense(layers, activation="relu", input_shape=(len(SHARED_MARKERS),)),
                     Dense(layers, activation="relu"),
                     Dense(1, activation="sigmoid")]
                )
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                model.fit(train_df_bootstrap, y_df_bootstrap)

                # Predicting on the original test set
                y_pred = model.predict(test_df)

                # convert output to binary
                y_pred = np.where(y_pred > 0.5, 1, 0)

                # calculate f1, recall, precision, accuracy
                f1 = f1_score(test_df_treatment, y_pred)
                recall = recall_score(test_df_treatment, y_pred)
                precision = precision_score(test_df_treatment, y_pred)

                print(
                    f"Results for protein {target_protein} and train treatment {treatment} and test treatment: {'ON' if treatment == 'PRE' else 'PRE'}")
                print(f"F1: {f1}")
                print(f"Recall: {recall}")
                print(f"Precision: {precision}")
                scores.append(
                    {"Train Treatment": treatment, "Protein": target_protein, "F1": f1, "Recall": recall,
                     "Precision": precision,
                     "Patient": patient, "Test Treatment": "ON" if treatment == "PRE" else "PRE"})

    # Saving results
    scores = pd.DataFrame(scores)
    scores.to_csv(Path(save_path, f"pre_post_scores.csv"), index=False)
