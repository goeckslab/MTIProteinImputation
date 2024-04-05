from pycaret.classification import ClassificationExperiment
from pathlib import Path
import pandas as pd
import os, sys, argparse

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER', "Treatment"]
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier", "pycaret")


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
        pre_treatment_path = Path("results", "ae_imputed_data", "single", mode, patient, f"{radius}",
                                  "pre_treatment.csv")
        on_treatment_path = Path("results", "ae_imputed_data", "single", mode, patient, f"{radius}", "on_treatment.csv")

        if not pre_treatment_path.exists() or not on_treatment_path.exists():
            print(f"Pre or on treatment path for patient {patient} does not exist.")
            print(f"Path: {pre_treatment_path}")
            print(f"Path: {on_treatment_path}")
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

    # select only the cells with a probability greater than 0.7
    correctly_predicted_cells_on = correctly_predicted_cells_on[correctly_predicted_cells_on["Probability"] > 0.4]
    correctly_predicted_cells_pre = correctly_predicted_cells_pre[correctly_predicted_cells_pre["Probability"] > 0.4]

    correctly_predicted_cell_index = pd.concat([correctly_predicted_cells_on, correctly_predicted_cells_pre]).index
    correctly_predicted_cells = test_data.loc[correctly_predicted_cell_index]
    correctly_predicted_cells_treatment = test_data_treatment.loc[correctly_predicted_cell_index]

    print(f"Selected {len(correctly_predicted_cells)} correctly predicted cells from {len(test_data)} cells.")
    return correctly_predicted_cells, correctly_predicted_cells_treatment


if __name__ == '__main__':

    exp = ClassificationExperiment()

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

    # assert that train data has no missing values
    assert train_data.isnull().sum().sum() == 0, "Train data has missing values"

    pre_treatment_test = pd.read_csv(Path("data", "bxs", f"{patient}_1.csv"))
    pre_treatment_test = clean_column_names(pre_treatment_test)
    pre_treatment_test["Treatment"] = "PRE"

    on_treatment_test = pd.read_csv(Path("data", "bxs", f"{patient}_2.csv"))
    on_treatment_test = clean_column_names(on_treatment_test)
    on_treatment_test["Treatment"] = "ON"
    test_data = pd.concat([pre_treatment_test, on_treatment_test])
    test_data = test_data[SHARED_MARKERS]
    # reset index
    test_data.reset_index(drop=True, inplace=True)

    # assert that test data has no missing values
    assert test_data.isnull().sum().sum() == 0, "Test data has missing values"

    imputed_data = load_imputed_data()
    assert len(imputed_data) == 4, "Imputed data should have 4 patients"

    # all patients except the patient
    train_imputed_data = pd.concat(
        [imputed_data[current_patient] for current_patient in imputed_data if current_patient != patient])
    test_imputed_data = imputed_data[patient]

    # check that train_imputed data shape and train_data_protein shape is similar
    assert train_imputed_data.shape == train_data.shape, "Train imputed data shape is not correct"

    test_data_treatment = test_data["Treatment"]
    # convert treatment to binary
    test_data_treatment = test_data_treatment.replace({"PRE": 0, "ON": 1})

    assert train_data.shape[1] == test_data.shape[
        1], f"Train and test data shape is not similar {train_data.shape} != {test_data.shape}"

    original_train_data = train_data.copy()
    original_test_data = test_data.copy()

    scores = []
    for target_protein in train_data.columns:
        train_data = original_train_data.copy()
        test_data = original_test_data.copy()

        if target_protein == "Treatment":
            continue
        print(f"Working on protein {target_protein}...")

        # Generating and training on bootstrap samples
        for i in range(30):
            og_experiment = ClassificationExperiment()
            og_experiment.setup(data=train_data, test_data=test_data, target="Treatment", session_id=42, index=False,
                                normalize=True, normalize_method="minmax", verbose=False)
            og_classifier = og_experiment.create_model("lightgbm")
            og_best = og_experiment.compare_models([og_classifier])

            og_predictions = og_experiment.predict_model(og_best, data=test_data)

            # pull f1 score from the model
            og_results = og_experiment.pull()
            # pull f1 score from the model
            og_score = og_results["Accuracy"].values[0]
            print(f"Score for protein {target_protein} using original data and bootstrap sample {i}: {og_score}")

            train_imp: pd.DataFrame = pd.DataFrame(train_data.copy(), columns=SHARED_MARKERS)
            # replace target protein with imputed data using loc
            train_imp.loc[:, target_protein] = train_imputed_data[target_protein].values

            test_imp: pd.DataFrame = pd.DataFrame(test_data.copy(), columns=SHARED_MARKERS)
            # replace target protein with imputed data using loc
            test_imp.loc[:, target_protein] = test_imputed_data[target_protein].values

            # check that remaining columns values are equal to original values
            assert train_imp.loc[:, train_data.columns != target_protein].equals(
                train_data.loc[:, train_data.columns != target_protein]), "Train data is not equal"

            assert test_imp.loc[:, test_data.columns != target_protein].equals(
                test_data.loc[:, test_data.columns != target_protein]), "Test data is not equal"

            imp_experiment = ClassificationExperiment()
            imp_experiment.setup(data=train_imp, test_data=test_imp, target="Treatment", session_id=42, index=False,
                                 normalize=True, normalize_method="minmax", verbose=False)
            imp_classifier = imp_experiment.create_model("lightgbm")
            imp_best = imp_experiment.compare_models([imp_classifier])

            imp_predictions = imp_experiment.predict_model(imp_best, data=test_imp)

            imp_results = imp_experiment.pull()
            # pull f1 score from the model
            imp_score = imp_results["Accuracy"].values[0]

            print(f"Score for protein {target_protein} using imputed data and bootstrap sample {i}: {imp_score}")

            # remove target protein from proteins
            rem_train = train_data.drop(columns=[target_protein])
            rem_test = test_data.drop(columns=[target_protein])

            remaining_markers = [marker for marker in SHARED_MARKERS if marker != target_protein]

            # check that x_train removed does not include the target proteins
            assert target_protein not in rem_train.columns, "Target protein is still in the data"
            assert target_protein not in rem_test.columns, "Target protein is still in the data"

            assert rem_train.shape[1] == rem_test.shape[1], "Train and test data shape is not similar"
            assert rem_train.shape[1] == train_data.shape[
                1] - 1, "Removed data and train data shape is not different-"

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
                f"Score for protein {target_protein} using data with removed protein and bootstrap sample {i}: {rem_score}")

            scores.append({"Protein": target_protein, "Imputed Score": imp_score, "Original Score": og_score,
                           "Removed Score": rem_score})
            # print current mean of scores for protein
            print(
                f"Mean scores for protein {target_protein} for {i + 1} runs:")
            print({pd.DataFrame(scores).groupby('Protein').mean()})

    scores = pd.DataFrame(scores)
    # calculate mean of proteins
    mean_scores = scores.groupby("Protein").mean().reset_index()
    mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
