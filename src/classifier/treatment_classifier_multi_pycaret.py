import argparse, sys, os
from pathlib import Path
import pandas as pd
from sklearn.utils import resample
import numpy as np
from pycaret.classification import ClassificationExperiment

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER', 'Treatment']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier_multi", "pycaret")


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
        pre_treatment_path = Path("results", "ae_imputed_data", "multi", mode, patient, f"{radius}",
                                  "pre_treatment.csv")
        on_treatment_path = Path("results", "ae_imputed_data", "multi", mode, patient, f"{radius}", "on_treatment.csv")

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
            data = data[SHARED_MARKERS]
            train_data.append(data)

    assert len(train_data) == 6, f"There should be 6 train datasets, loaded {len(train_data)}"
    return pd.concat(train_data)


def get_correctly_confident_cells(model, probabilities, test_data, treatment_data):
    predictions = model.predict(test_data)

    confidence = pd.DataFrame(columns=["Probability", "Treatment", "Predictions"])
    confidence["Treatment"] = treatment_data
    # First, ensure 'confidence' DataFrame has a column to store probabilities
    if 'Probability' not in confidence.columns:
        confidence['Probability'] = 0  # Initializes the column with zeros

    # Vectorized assignment using numpy where
    # Update 'Probability' based on 'test_data_treatment' values and corresponding 'og_probs'
    confidence['Probability'] = np.where(treatment_data == "ON", probabilities[:, 1], probabilities[:, 0])
    # confidence.reset_index(drop=True, inplace=True)
    confidence["Predictions"] = predictions

    # select correctly predicted cells compared to the test_treatment data
    correctly_predicted_cells_on = confidence[(confidence["Predictions"] == "ON") & (
            test_data_treatment == "ON")]
    correctly_predicted_cells_pre = confidence[(confidence["Predictions"] == "PRE") & (
            test_data_treatment == "PRE")]

    correctly_predicted_cell_index = pd.concat([correctly_predicted_cells_on, correctly_predicted_cells_pre]).index
    correctly_predicted_cells = test_data.loc[correctly_predicted_cell_index]
    correctly_predicted_cells_treatment = test_data_treatment.loc[correctly_predicted_cell_index]

    print(f"Selected {len(correctly_predicted_cells)} correctly predicted cells from {len(test_data)} cells.")
    return correctly_predicted_cells, correctly_predicted_cells_treatment


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
    train_data.reset_index(drop=True, inplace=True)

    pre_treatment_test = pd.read_csv(Path("data", "bxs", f"{patient}_1.csv"))
    pre_treatment_test = clean_column_names(pre_treatment_test)
    pre_treatment_test["Treatment"] = "PRE"

    on_treatment_test = pd.read_csv(Path("data", "bxs", f"{patient}_2.csv"))
    on_treatment_test = clean_column_names(on_treatment_test)
    on_treatment_test["Treatment"] = "ON"
    test_data = pd.concat([pre_treatment_test, on_treatment_test])
    test_data = test_data[SHARED_MARKERS]
    test_data.reset_index(drop=True, inplace=True)

    train_data_treatment = train_data["Treatment"]
    test_data_treatment = test_data["Treatment"]

    imputed_data = load_imputed_data()
    assert len(imputed_data) == 4, "Imputed data should have 4 patients"

    # all patients except the patient
    train_imputed_data = pd.concat(
        [imputed_data[current_patient] for current_patient in imputed_data if current_patient != patient])
    test_imputed_data = imputed_data[patient]

    # check that train_imputed data shape and train_data_protein shape is similar
    assert train_imputed_data.shape == train_data.shape, "Train imputed data shape is not correct"

    # load rounds
    rounds = pd.read_csv(Path("data", "rounds", "rounds.csv"))
    # select rounds for the given patient
    rounds = rounds[rounds["Patient"] == patient.replace("_", " ")]

    scores = []

    original_train_data = train_data.copy()
    original_test_data = test_data.copy()

    for round in rounds["Round"].unique():
        train_data = original_train_data.copy()
        test_data = original_test_data.copy()

        print(f"Working on round {round}...")
        target_proteins = rounds[rounds["Round"] == round]["Marker"].to_list()

        # adjust the target proteins to leave only the shared markers in the list
        target_proteins = [protein for protein in target_proteins if protein in SHARED_MARKERS]
        if len(target_proteins) == 0:
            print(f"No shared markers found for round {round}. Skipping...")
            continue

        print("Found proteins: ", target_proteins)

        # Generating and training on bootstrap samples
        for i in range(30):
            og_experiment = ClassificationExperiment()
            og_experiment.setup(data=train_data, test_data=test_data, target="Treatment", session_id=42, index=False,
                                normalize=True, normalize_method="minmax", verbose=True)
            og_classifier = og_experiment.create_model("lightgbm")
            og_best = og_experiment.compare_models([og_classifier])

            og_predictions = og_experiment.predict_model(og_best, data=test_data)

            # pull f1 score from the model
            og_results = og_experiment.pull()
            # pull f1 score from the model
            og_score = og_results["Accuracy"].values[0]

            print(f"Score for proteins {target_proteins} using bootstrap sample {i}: {og_score}")

            # replace target protein with imputed data using loc
            train_imp: pd.DataFrame = pd.DataFrame(train_data.copy(), columns=SHARED_MARKERS)
            # replace target protein with imputed data using loc
            train_imp.loc[:, target_proteins] = train_imputed_data[target_proteins].values

            test_imp: pd.DataFrame = pd.DataFrame(test_data.copy(), columns=SHARED_MARKERS)
            # replace target protein with imputed data using loc
            test_imp.loc[:, target_proteins] = test_imputed_data[target_proteins].values

            # select only proteins that are not in target proteins
            remaining_markers = [marker for marker in SHARED_MARKERS if marker not in target_proteins]

            # check that all remaining columns values are equal to original values
            assert train_imp.loc[:, remaining_markers].equals(
                train_data.loc[:, remaining_markers]), "Train data is not equal"

            assert test_imp.loc[:, remaining_markers].equals(
                test_data.loc[:, remaining_markers]), "Test data is not equal"

            imp_experiment = ClassificationExperiment()
            imp_experiment.setup(data=train_imp, test_data=test_imp, target="Treatment", session_id=42, index=False,
                                 normalize=True, normalize_method="minmax", verbose=False)
            imp_classifier = imp_experiment.create_model("lightgbm")
            imp_best = imp_experiment.compare_models([imp_classifier])

            imp_predictions = imp_experiment.predict_model(imp_best, data=test_imp)

            imp_results = imp_experiment.pull()
            # pull f1 score from the model
            imp_score = imp_results["Accuracy"].values[0]

            print(f"Score for proteins {target_proteins} using imputed data and bootstrap sample {i}: {imp_score}")

            # remove target protein from proteins
            rem_train = train_data.drop(columns=target_proteins)
            rem_test = test_data.drop(columns=target_proteins)

            remaining_markers = [marker for marker in SHARED_MARKERS if marker not in target_proteins]

            # check that x_train removed does not include the target proteins
            assert not any([protein in rem_train.columns for protein in
                            target_proteins]), "Target protein is still in the data"
            assert not any([protein in rem_test.columns for protein in
                            target_proteins]), "Target protein is still in the data"

            assert rem_train.shape[1] == rem_test.shape[1], "Train and test data shape is not similar"
            assert rem_train.shape[1] == original_train_data.shape[
                1] - len(target_proteins), "Removed data and train data shape is not different"

            rem_experiment = ClassificationExperiment()
            rem_experiment.setup(data=rem_train, test_data=rem_test, target="Treatment", session_id=42,
                                 index=False, normalize=True, normalize_method="minmax", verbose=False)
            rem_classifier = rem_experiment.create_model("lightgbm")
            rem_best = rem_experiment.compare_models([rem_classifier])

            rem_predictions = rem_experiment.predict_model(rem_best, data=rem_test)

            rem_results = rem_experiment.pull()
            # pull f1 score from the model
            rem_score = rem_results["Accuracy"].values[0]

            print(
                f"Score for proteins {target_proteins} using data with removed protein and bootstrap sample {i}: {rem_score}")

            # add each protein to the scores list in a single column

            scores.append({"Round": round, "Imputed Score": imp_score, "Original Score": og_score,
                           "Removed Score": rem_score, "Iteration": i})
            # print current mean of scores for protein
            print(
                f"Mean scores for protein {target_proteins} in round {round}: {pd.DataFrame(scores).groupby('Round').mean()}")

    scores = pd.DataFrame(scores)
    # calculate mean of proteins
    mean_scores = scores.groupby("Round").mean().reset_index()
    mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
