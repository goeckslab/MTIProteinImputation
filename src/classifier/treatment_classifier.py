import argparse, sys, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier")


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


def load_imputed_data() -> dict:
    imputed_data = {}
    for patient in PATIENTS:
        pre_treatment_path = Path("data", "ae_reconstructed_data", mode, patient, f"{radius}", "pre_treatment.csv")
        on_treatment_path = Path("data", "ae_reconstructed_data", mode, patient, f"{radius}", "on_treatment.csv")

        if not pre_treatment_path.exists() or not on_treatment_path.exists():
            print(f"Pre or on treatment file for {patient} does not exist.")
            sys.exit(1)

        # load files
        pre_treatment = pd.read_csv(pre_treatment_path)
        on_treatment = pd.read_csv(on_treatment_path)

        # combine data
        combined_data = pd.concat([pre_treatment, on_treatment])
        imputed_proteins = combined_data[SHARED_MARKERS]
        imputed_data[patient] = imputed_proteins
    return imputed_data


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', "-r", type=int, default=0, choices=[0, 15, 30, 60, 90, 120])
    parser.add_argument("--patient", "-p", type=str, required=True, help="Patient data to use")
    parser.add_argument("--mode", "-m", type=str, required=True, choices=["ip", "exp"], default="ip")

    args = parser.parse_args()

    radius: int = args.radius
    patient: str = args.patient
    mode: str = args.mode

    print(f"Using radius: {radius} µm")
    print(f"Using patient: {patient}")
    print(f"Using mode: {mode}")

    save_path = Path(save_path, mode, patient, str(radius))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    train_data = load_train_data(Path("data", "bxs"), patient)

    pre_treatment_test = pd.read_csv(Path("data", "bxs", f"{patient}_1.csv"))
    pre_treatment_test = clean_column_names(pre_treatment_test)
    pre_treatment_test["Treatment"] = "PRE"

    on_treatment_test = pd.read_csv(Path("data", "bxs", f"{patient}_2.csv"))
    on_treatment_test = clean_column_names(on_treatment_test)
    on_treatment_test["Treatment"] = "ON"
    test_data = pd.concat([pre_treatment_test, on_treatment_test])

    train_data_proteins = train_data[SHARED_MARKERS]
    test_data_proteins = test_data[SHARED_MARKERS]

    train_data_treatment = train_data["Treatment"]
    test_data_treatment = test_data["Treatment"]

    imputed_data = load_imputed_data()
    assert len(imputed_data) == 4, "Imputed data should have 4 patients"

    # all patients except the patient
    train_imputed_data = pd.concat(
        [imputed_data[current_patient] for current_patient in imputed_data if current_patient != patient])
    test_imputed_data = imputed_data[patient]

    # check that train_imputed data shape and train_data_protein shape is similar
    assert train_imputed_data.shape == train_data_proteins.shape, "Train imputed data shape is not correct"

    scores = []
    for target_protein in test_data_proteins.columns:
        print(f"Working on protein {target_protein}...")

        # scale the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_data_proteins)
        X_test = scaler.transform(test_data_proteins)

        # train classifier on masked data using random forest using sklearn
        model = DecisionTreeClassifier()
        model.fit(X_train, train_data_treatment)
        original_score = model.score(X_test, test_data_treatment)
        print(f"Score for protein {target_protein} using original data: {original_score}")

        train_data_proteins[target_protein] = train_imputed_data[target_protein].values

        # check that train data target protein values are different from the original train data
        assert not train_data_proteins[target_protein].equals(train_data[target_protein]), \
            "Train data target protein values are the same as the original train data"

        # scale the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_data_proteins)
        X_test = scaler.transform(test_data_proteins)

        # train classifier on masked data using random forest using sklearn
        model = DecisionTreeClassifier()
        model.fit(X_train, train_data_treatment)
        imputed_score = model.score(X_test, test_data_treatment)
        print(f"Score for protein {target_protein} using imputed data: {imputed_score}")

        # remove target protein from proteins
        X_train = train_data_proteins.drop(columns=[target_protein])
        X_test = test_data_proteins.drop(columns=[target_protein])

        # shape of X_train and X_test should be one less that the original_proteins
        assert X_train.shape[1] == train_data_proteins.shape[1] - 1, "X_train shape is not correct"
        assert X_test.shape[1] == test_data_proteins.shape[1] - 1, "X_test shape is not correct"
        # check that in x train only SHARED_MARKERS are present
        assert all([protein in SHARED_MARKERS for protein in X_train.columns]), "X_train contains other proteins"

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # train classifier on original data using random forest using sklearn
        model = RandomForestClassifier()
        model.fit(X_train, train_data_treatment)
        removed_score = model.score(X_test, test_data_treatment)

        print(f"Score for protein {target_protein} using data with removed protein: {removed_score}")

        # save the scores
        scores.append({"Protein": target_protein, "Imputed Score": imputed_score, "Original Score": original_score,
                       "Removed score": removed_score})

    scores = pd.DataFrame(scores)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
