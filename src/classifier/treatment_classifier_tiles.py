from pycaret.classification import ClassificationExperiment
from pathlib import Path
import pandas as pd
import os, sys, argparse
import numpy as np

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

    assert len(imputed_data) == 4, "Imputed data should have 4 patients"
    return imputed_data


def load_train_data(base_path: Path, patient: str) -> {}:
    train_data_sets = {}
    for file in os.listdir(base_path):
        file_name = Path(file).stem
        if file.endswith(".csv") and patient not in file_name:
            print("Loading train file: " + file)
            data = pd.read_csv(Path(base_path, file))
            data = clean_column_names(data)

            end = file_name.split("_")[-1]
            end = f"_{end}"

            data["Treatment"] = "PRE" if "_1" in end else "ON"

            if file == "9_14_2.csv" or file == "9_15_2.csv":
                assert data["Treatment"].values[0] == "ON", "Treatment should be ON for patient 9_14_2"
            elif file == "9_14_1.csv" or file == "9_15_1.csv":
                assert data["Treatment"].values[0] == "PRE", "Treatment should be PRE for patient 9_14_1"

            assert "Treatment" in data.columns, f"Treatment column is missing for dataframe of patient {file}"
            # merge SHARED MARKER with X and Y centroid
            shared_columns = SHARED_MARKERS + ["X_centroid", "Y_centroid"]
            data = data[shared_columns]

            train_data_sets[Path(file).stem].append(data)

    assert len(train_data_sets) == 6, f"There should be 6 train datasets, loaded {len(train_data_sets)}"
    return train_data_sets


def get_non_confident(predicted_data: pd.DataFrame, remove_marker: str = None):
    # get all cells with prediction scores between 0.3 and 0.7
    confident_cells = predicted_data[
        (predicted_data["prediction_score"] >= 0.4) & (predicted_data["prediction_score"] <= 0.6)]

    # get all high confident cells
    # confident_cells = predicted_data[predicted_data["prediction_score"] >= 0.8]
    if remove_marker:
        # remove the marker from SHARED_MARKER list
        rem_markers = [marker for marker in SHARED_MARKERS if marker != remove_marker]
        return confident_cells[rem_markers]

    return confident_cells[SHARED_MARKERS]


def create_tile_dataset(df: pd.DataFrame, tile_size, amount_of_tiles=100, x_col='X_centroid', y_col='Y_centroid') -> []:
    """
    Extracts a random tile from a pandas DataFrame containing x and y coordinates.

    Parameters:
    - df: pandas DataFrame, contains the dataset with 'X' and 'Y' columns for coordinates.
    - tile_size: float, the size of the tile to extract.
    - x_col: str, the name of the column containing the x coordinates.
    - y_col: str, the name of the column containing the y coordinates.

    Returns:
    - tile_df: pandas DataFrame, the randomly selected tile of data.
    """

    # Calculate the bounds for your tiles based on the entire biopsy
    x_min, x_max = df['X_centroid'].min(), df['X_centroid'].max() - tile_size
    y_min, y_max = df['Y_centroid'].min(), df['Y_centroid'].max() - tile_size

    # Initialize a list to hold the top-left coordinates and data points for each tile
    tile_data = []
    max_attempts = 300

    # Randomly choose 100 starting points for the tiles
    for _ in range(amount_of_tiles):
        # Initialize an empty DataFrame for the tile
        tile_df = pd.DataFrame()
        counter = 0
        while tile_df.empty and counter < max_attempts:
            x_start = np.random.uniform(x_min, x_max)
            y_start = np.random.uniform(y_min, y_max)
            # Define the boundaries of the tile
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Extract the cells within this tile
            tile_df = df[(df['X_centroid'] >= x_start) &
                         (df['X_centroid'] < x_end) &
                         (df['Y_centroid'] >= y_start) &
                         (df['Y_centroid'] < y_end)]

            # Calculate features for this tile
            if tile_df.empty:
                counter += 1
                continue

            features = pd.DataFrame(tile_df.mean()).T
            # calculate the number of cells in the tile
            features["cell_count"] = len(tile_df)

            # add the std for each column of the SHARED COLUMNS
            for col in SHARED_MARKERS:
                if col == "Treatment":
                    continue
                features[f"{col}_std"] = tile_df[col].std()

            # add the treatment
            features["Treatment"] = df["Treatment"].values[0]

            # remove Patient id
            if "Patient Id" in features.columns:
                features.drop("Patient Id", axis=1, inplace=True)

            # assert that no patient id or cell type column are in the dataset
            assert "Patient Id" not in features.columns, "Patient Id column is in the dataset"
            assert "cell_type" not in features.columns, "Cell type column is in the dataset"

            # check that treatment are in the dataset
            assert "Treatment" in features.columns, "Treatment column is missing"

            tile_data.append(features)

    # Convert tile data into a DataFrame
    tiles_df = pd.concat(tile_data, axis=0)
    # fill NaN values with 0
    tiles_df.fillna(0, inplace=True)
    return tiles_df


def load_test_data() -> {}:
    test_data_sets = {}
    pre_biopsy: Path = Path(f"{patient}_1.csv")
    pre_treatment_test = pd.read_csv(Path("data", "bxs", pre_biopsy))
    pre_treatment_test = clean_column_names(pre_treatment_test)
    pre_treatment_test["Treatment"] = "PRE"
    assert len(pre_treatment_test) > 0, "Pre treatment test data is empty"
    test_data_sets[pre_biopsy] = pre_treatment_test

    on_biopsy: Path = Path(f"{patient}_2.csv")
    on_treatment_test = pd.read_csv(Path("data", "bxs", on_biopsy))
    on_treatment_test = clean_column_names(on_treatment_test)
    on_treatment_test["Treatment"] = "ON"
    assert len(on_treatment_test) > 0, "On treatment test data is empty"
    test_data_sets[on_biopsy] = on_treatment_test

    return test_data_sets


def create_tiles_for_dfs(dataframes: [pd.DataFrame], tile_size: int, amount_of_tiles: int) -> pd.DataFrame:
    # create tiles for original data
    tile_sets = []
    for data_set in dataframes:
        tile_set = create_tile_dataset(data_set, tile_size, amount_of_tiles)
        tile_sets.append(tile_set)

    tile_set = pd.concat(tile_sets)
    tile_set.reset_index(drop=True, inplace=True)
    return tile_set


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

    # load train data sets
    train_data_sets: dict = load_train_data(Path("data", "bxs"), patient)
    # load test data sets
    test_data_sets: {} = load_test_data()
    # load imputed data
    imputed_data: {} = load_imputed_data()

    scores = []
    # prepare tile datasets for original, imputed and removed data
    for target_protein in SHARED_MARKERS:
        if target_protein == "Treatment":
            continue

        for i in range(30):

            og_tile_train_set = create_tiles_for_dfs(train_data_sets.values(), 200, 100)
            og_tile_test_set = create_tiles_for_dfs(test_data_sets.values(), 200, 100)

            adjusted_imputed_train_data = {}
            # create new imp dataset from original data and replace the target protein with the imputated proteins value by train patients
            for biopsy in train_data_sets.keys():
                sub_train_data = train_data_sets[biopsy].copy()
                sub_imp_train_data = imputed_data[biopsy]
                # replace target protein with imputed data using the indexes of the subset
                sub_train_data.loc[:, target_protein] = sub_imp_train_data.loc[:, target_protein]
                # assert that sub train data is not equal to original data
                assert not sub_train_data.equals(train_data_sets[biopsy]), "Sub train data is equal to original data"
                adjusted_imputed_train_data[biopsy] = sub_imp_train_data.copy()

            adjusted_imputed_test_data = {}
            for biopsy in test_data_sets.keys():
                sub_test_data = test_data_sets[biopsy].copy()
                sub_imp_test_data = imputed_data[biopsy]

                sub_test_data.loc[:, target_protein] = sub_imp_test_data.loc[:, target_protein]

                assert not sub_test_data.equals(test_data_sets[biopsy]), "Sub train data is equal to original data"
                adjusted_imputed_test_data[biopsy] = sub_imp_test_data

            imp_tile_train_set: pd.DataFrame = create_tiles_for_dfs(adjusted_imputed_train_data.values(), 200, 100)
            imp_tile_test_set: pd.DataFrame = create_tiles_for_dfs(adjusted_imputed_test_data.values(), 200, 100)

            rem_train_sets = {}
            for biopsy in train_data_sets.keys():
                rem_train_sets[biopsy] = train_data_sets[biopsy].copy()
                rem_train_sets[biopsy].drop(columns=[target_protein], inplace=True)

            rem_test_sets = {}
            for biopsy in test_data_sets.keys():
                rem_test_sets[biopsy] = test_data_sets[biopsy].copy()
                rem_test_sets[biopsy].drop(columns=[target_protein], inplace=True)

            rem_tile_train_set: pd.DataFrame = create_tiles_for_dfs(rem_train_sets.values(), 200, 100)
            rem_tile_test_set: pd.DataFrame = create_tiles_for_dfs(rem_test_sets.values(), 200, 100)

            # check that x_train removed does not include the target proteins
            assert target_protein not in rem_tile_train_set.columns, "Target protein is still in the data"
            assert target_protein not in rem_tile_test_set.columns, "Target protein is still in the data"

            assert rem_tile_train_set.shape[1] == rem_tile_test_set.shape[1], "Train and test data shape is not similar"
            assert rem_tile_train_set.shape[1] == og_tile_train_set.shape[
                1] - 1, "Removed data and train data shape is not different-"


            # run experiments
            og_experiment = ClassificationExperiment()
            og_experiment.setup(data=og_tile_train_set, target="Treatment", session_id=42,
                                index=True, normalize=True, normalize_method="minmax", verbose=True, fold=10, fold_shuffle=True)
            og_classifier = og_experiment.create_model("lightgbm", verbose=False)
            og_best = og_experiment.compare_models([og_classifier], verbose=False)
            og_predictions = og_experiment.predict_model(og_best, data=og_tile_test_set, verbose=False)

            og_results = og_experiment.pull()
            # pull f1 score from the model
            og_score = og_results["Accuracy"].values[0]
            print(f"Score for protein {target_protein} using original data and bootstrap sample {i}: {og_score}")

            imp_experiment = ClassificationExperiment()
            imp_experiment.setup(data=imp_tile_train_set, target="Treatment", session_id=42, index=True,
                                 normalize=True, normalize_method="minmax", verbose=False)
            imp_classifier = imp_experiment.create_model("lightgbm", verbose=False)
            imp_best = imp_experiment.compare_models([imp_classifier], verbose=False)

            imp_predictions = imp_experiment.predict_model(imp_best, data=imp_tile_test_set, verbose=False)
            imp_subset = get_non_confident(imp_predictions)
            imp_predictions = imp_experiment.predict_model(imp_best, data=imp_subset, verbose=False)

            imp_results = imp_experiment.pull()
            # pull the score from the model
            imp_score = imp_results["Accuracy"].values[0]

            print(f"Score for protein {target_protein} using imputed data and bootstrap sample {i}: {imp_score}")

            rem_experiment = ClassificationExperiment()
            rem_experiment.setup(data=rem_tile_train_set,  target="Treatment", session_id=42,
                                 index=True, normalize=True, normalize_method="minmax", verbose=True)
            rem_classifier = rem_experiment.create_model("lightgbm", verbose=False)
            rem_best = rem_experiment.compare_models([rem_classifier], verbose=False)

            rem_predictions = rem_experiment.predict_model(rem_best, data=rem_tile_test_set, verbose=False)
            rem_subset = get_non_confident(rem_predictions, remove_marker=target_protein)
            rem_predictions = rem_experiment.predict_model(rem_best, data=rem_subset, verbose=False)

            rem_results = rem_experiment.pull()
            # pull the score from the model
            rem_score = rem_results["Accuracy"].values[0]

            print(
                f"Score for protein {target_protein} using data with removed protein and bootstrap sample {i}: {rem_score}")

            scores.append({"Protein": target_protein, "Imputed Score": imp_score, "Original Score": og_score,
                           "Removed Score": rem_score})
            # print current mean of scores for protein
            print(
                f"Mean scores for protein {target_protein} for {i + 1} runs:")
            print(pd.DataFrame(scores).groupby('Protein').mean())

    scores = pd.DataFrame(scores)
    # calculate mean of proteins
    mean_scores = scores.groupby("Protein").mean().reset_index()
    mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)


