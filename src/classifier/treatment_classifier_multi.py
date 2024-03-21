import argparse, sys, os
from pathlib import Path
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier_multi")


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


def classifier(X_train, y_train, X_test, y_test):
    # Creating a bootstrap sample of the training data
    X_train_boot, y_train_boot = resample(X_train, y_train)

    # Train classifier on the bootstrap sample
    model = DecisionTreeClassifier()
    model.fit(X_train_boot, y_train_boot)
    return model.score(X_test, y_test)


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

    # load rounds
    rounds = pd.read_csv(Path("data", "rounds", "rounds.csv"))
    # select rounds for the given patient
    rounds = rounds[rounds["Patient"] == patient.replace("_", " ")]

    scores = []
    for round in rounds["Round"].unique():
        print(f"Working on round {round}...")
        target_proteins = rounds[rounds["Round"] == round]["Marker"].to_list()

        # adjust the target proteins to leave only the shared markers in the list
        target_proteins = [protein for protein in target_proteins if protein in SHARED_MARKERS]
        if len(target_proteins) == 0:
            print(f"No shared markers found for round {round}. Skipping...")
            continue

        print("Found proteins: ", target_proteins)

        # scale the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_data_proteins)
        X_test = scaler.transform(test_data_proteins)

        # Generating and training on bootstrap samples
        for i in range(30):
            original_score = classifier(X_train, train_data_treatment, X_test, test_data_treatment)
            print(f"Score for proteins {target_proteins} using bootstrap sample {i}: {original_score}")

            # replace target protein with imputed data using loc
            train_data_proteins.loc[:, target_proteins] = train_imputed_data[target_proteins].values

            # assert that values of target proteins are different to original dataset
            assert not train_data_proteins.equals(train_imputed_data), "Train data proteins are the same as imputed data"

            X_train = scaler.fit_transform(train_data_proteins)
            imputed_score = classifier(X_train, train_data_treatment, X_test, test_data_treatment)

            print(f"Score for proteins {target_proteins} using imputed data and bootstrap sample {i}: {imputed_score}")

            # remove target protein from proteins
            X_train_removed = train_data_proteins.drop(columns=target_proteins)
            X_test_removed = test_data_proteins.drop(columns=target_proteins)

            assert X_train_removed.shape[1] == X_test_removed.shape[1], "Train and test data shape is not similar"
            assert X_train_removed.shape[1] == train_data_proteins.shape[
                1] - len(target_proteins), "Removed data and train data shape is not different-"

            removed_score = classifier(X_train_removed, train_data_treatment, X_test_removed, test_data_treatment)

            print(
                f"Score for protein {target_proteins} using data with removed protein and bootstrap sample {i}: {removed_score}")

            # add each protein to the scores list in a single column

            scores.append({"Round": round, "Imputed Score": imputed_score, "Original Score": original_score,
                           "Removed Score": removed_score})
            # print current mean of scores for protein
            print(f"Mean scores for protein {target_proteins}: {pd.DataFrame(scores).groupby('Round').mean()}")

    scores = pd.DataFrame(scores)
    # calculate mean of proteins
    mean_scores = scores.groupby("Protein").mean().reset_index()
    mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
