from pycaret.classification import ClassificationExperiment
from pathlib import Path
import pandas as pd
import os, sys, argparse
import numpy as np
from tqdm import tqdm

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER', "Treatment"]
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier", "pycaret_tiles")


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
        pre_treatment = pre_treatment[SHARED_MARKERS]
        imputed_data[f"{patient}_1"] = pre_treatment

        on_treatment = pd.read_csv(on_treatment_path)
        on_treatment = on_treatment[SHARED_MARKERS]
        imputed_data[f"{patient}_2"] = on_treatment

    assert len(imputed_data) == 8, f"Imputed data should have 8 biopsies, had {len(imputed_data)}"
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

            train_data_sets[Path(file).stem] = data

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


def calculate_neighbor_stats(df, cell_radius=30):
    # Define the search range
    x_range = pd.IntervalIndex.from_arrays(df['X_centroid'] - cell_radius, df['X_centroid'] + cell_radius,
                                           closed='both')
    y_range = pd.IntervalIndex.from_arrays(df['Y_centroid'] - cell_radius, df['Y_centroid'] + cell_radius,
                                           closed='both')

    # Initialize containers for results
    mean_results = []
    std_results = []

    # drop treatment x, and y centroid columns

    # Iterate over intervals
    for interval in range(len(df)):
        # Find neighbors within the x and y range
        neighbors = df[x_range.overlaps(x_range[interval]) & y_range.overlaps(y_range[interval])]

        # Calculate mean and std
        mean_results.append(neighbors.mean())
        std_results.append(neighbors.std())

    # Convert list of Series to DataFrame
    means_df = pd.DataFrame(mean_results)
    means_df = means_df[SHARED_MARKERS[:-1]]
    # rename columns of means_df to neighbor_mean_{column}
    means_df.columns = [f'neighbor_mean_{col}' for col in means_df.columns]

    stds_df = pd.DataFrame(std_results)
    stds_df = stds_df[SHARED_MARKERS[:-1]]
    # rename columns of stds_df to neighbor_std_{column}
    stds_df.columns = [f'neighbor_std_{col}' for col in stds_df.columns]

    # add means df columns to the original dataframe using loc
    df.loc[:, means_df.columns] = means_df.values
    # add stds df columns to the original dataframe using loc
    df.loc[:, stds_df.columns] = stds_df.values

    # fill NaN values with 0
    df.fillna(0, inplace=True)

    # assert that no NAN values are in the dataframe
    assert not df.isnull().values.any(), "NAN values are in the dataset"
    return df


def create_tile_dataset(df: pd.DataFrame, tile_size, amount_of_tiles=100, x_col='X_centroid', y_col='Y_centroid',
                        removed_protein: str = '') -> []:
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

    # Randomly choose 100 starting points for the tiles
    # Assuming you define x_min, x_max, y_min, y_max based on your DataFrame df.
    x_min, x_max = df[x_col].min(), df[x_col].max() - tile_size
    y_min, y_max = df[y_col].min(), df[y_col].max() - tile_size

    # Initialize a list to hold each tile's DataFrame after processing
    tile_data = []
    max_attempts = 300

    # Randomly choose starting points for the tiles
    for _ in range(amount_of_tiles):
        counter = 0
        successful_tile = False
        while not successful_tile and counter < max_attempts:
            x_start = np.random.uniform(x_min, x_max)
            y_start = np.random.uniform(y_min, y_max)
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Extract the cells within this tile
            tile_df = df[(df[x_col] >= x_start) & (df[x_col] < x_end) &
                         (df[y_col] >= y_start) & (df[y_col] < y_end)]

            # If the tile has less than 200 cells, retry
            if tile_df.empty or len(tile_df) < 200:
                counter += 1
            else:
                successful_tile = True
                features = pd.DataFrame(tile_df.mean()).T
                # calculate the number of cells in the tile
                features["cell_count"] = len(tile_df)

                features = calculate_neighbor_stats(tile_df)

                for col in SHARED_MARKERS:
                    if col == "Treatment":
                        continue

                    if col == removed_protein:
                        continue
                    # add the mean for each column of the SHARED COLUMNS
                    features[f"{col}_mean"] = tile_df[col].mean()
                    # add the std for each column of the SHARED COLUMNS
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
                # remove the x and y centroid
                features.drop(["X_centroid", "Y_centroid"], axis=1, inplace=True)

                tile_data.append(features)

    # Convert tile data into a DataFrame
    tiles_df = pd.concat(tile_data, axis=0)

    # fill NaN values with 0
    tiles_df.fillna(0, inplace=True)
    return tiles_df


def load_test_data() -> {}:
    test_data_sets = {}
    pre_biopsy_path: Path = Path(f"{patient}_1.csv")
    pre_treatment_test = pd.read_csv(Path("data", "bxs", pre_biopsy_path))
    pre_treatment_test = clean_column_names(pre_treatment_test)
    pre_treatment_test["Treatment"] = "PRE"
    pre_treatment_test = pre_treatment_test[SHARED_MARKERS + ["X_centroid", "Y_centroid"]]
    assert len(pre_treatment_test) > 0, "Pre treatment test data is empty"
    test_data_sets[Path(pre_biopsy_path).stem] = pre_treatment_test

    on_biopsy_path: Path = Path(f"{patient}_2.csv")
    on_treatment_test = pd.read_csv(Path("data", "bxs", on_biopsy_path))
    on_treatment_test = clean_column_names(on_treatment_test)
    on_treatment_test["Treatment"] = "ON"
    on_treatment_test = on_treatment_test[SHARED_MARKERS + ["X_centroid", "Y_centroid"]]
    assert len(on_treatment_test) > 0, "On treatment test data is empty"
    test_data_sets[Path(on_biopsy_path).stem] = on_treatment_test

    return test_data_sets


def create_tiles_for_dfs(dataframes: [pd.DataFrame], tile_size: int, amount_of_tiles: int,
                         removed_protein: str = '') -> pd.DataFrame:
    # create tiles for original data
    tile_sets = []
    for data_set in dataframes:
        tile_set = create_tile_dataset(data_set, tile_size, amount_of_tiles, removed_protein=removed_protein)
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
    parser.add_argument("--tile_size", "-ts", type=int, default=200, help="Tile size to use")
    args = parser.parse_args()

    radius: int = args.radius
    patient: str = args.patient
    mode: str = args.mode
    tile_size: int = args.tile_size

    print(f"Using radius: {radius} µm")
    print(f"Using patient: {patient}")
    print(f"Using mode: {mode}")
    print(f"Using tile size: {tile_size}")

    save_path = Path(save_path, mode, patient, str(radius))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load train data sets
    loaded_train_data_sets: {} = load_train_data(Path("data", "bxs"), patient)
    # load test data sets
    loaded_test_data_sets: {} = load_test_data()
    # load imputed data
    loaded_imputed_data: {} = load_imputed_data()

    scores = []
    # prepare tile datasets for original, imputed and removed data
    for target_protein in SHARED_MARKERS:
        if target_protein == "Treatment":
            continue

        print(f"Running protein {target_protein}...")

        og_tile_train_sets = {}
        og_tile_test_sets = {}

        imp_tile_train_sets = {}
        imp_tile_test_sets = {}

        rem_tile_train_sets = {}
        rem_tile_test_sets = {}
        print("Preparing tiles for data...")
        for i in tqdm(range(30)):
            train_data_sets = {}
            test_data_sets = {}

            # copy each dataset in the dictionary
            for key in loaded_train_data_sets.keys():
                train_data_sets[key] = loaded_train_data_sets[key].copy()

            for key in loaded_test_data_sets.keys():
                test_data_sets[key] = loaded_test_data_sets[key].copy()

            og_tile_train_set = create_tiles_for_dfs(train_data_sets.values(), tile_size, 100)

            # treatment should be both pre and on
            assert len(og_tile_train_set["Treatment"].unique()) == 2, "Treatment should be both pre and on"

            og_tile_test_set = create_tiles_for_dfs(test_data_sets.values(), tile_size, 100)
            assert len(og_tile_test_set["Treatment"].unique()) == 2, "Treatment should be both pre and on"

            # assert that target protein is in train and test data
            assert target_protein in og_tile_train_set.columns, "Target protein is not in the train data"
            assert target_protein in og_tile_test_set.columns, "Target protein is not in the test data"

            og_tile_train_sets[i] = og_tile_train_set
            og_tile_test_sets[i] = og_tile_test_set

            adjusted_imputed_train_data = {}
            # create new imp dataset from original data and replace the target protein with the imputated proteins value by train patients
            for biopsy in train_data_sets.keys():
                sub_train_data = train_data_sets[biopsy].copy()
                sub_imp_train_data = loaded_imputed_data[biopsy].copy()
                # replace target protein with imputed data using the indexes of the subset
                sub_train_data.loc[:, target_protein] = sub_imp_train_data.loc[:, target_protein]

                #  assert that target protein is not in the original data
                assert not sub_train_data[target_protein].equals(
                    train_data_sets[biopsy][target_protein]), "Target protein train data is equal to original data"

                # assert that except of the target protein the data is the same
                assert sub_train_data.drop(columns=[target_protein]).equals(
                    train_data_sets[biopsy].drop(
                        columns=[target_protein])), "Data is not the same except for the target protein"

                adjusted_imputed_train_data[biopsy] = sub_train_data.copy()

            adjusted_imputed_test_data = {}
            for biopsy in test_data_sets.keys():
                sub_test_data = test_data_sets[biopsy].copy()
                sub_imp_test_data = loaded_imputed_data[biopsy].copy()
                sub_test_data.loc[:, target_protein] = sub_imp_test_data.loc[:, target_protein]

                # assert that target proteins of subtest data is different compared to test_data_set
                assert not sub_test_data[target_protein].equals(
                    test_data_sets[biopsy][target_protein]), "Target protein test data is equal to original data"

                # assert that except of the target protein the data is the same
                assert sub_test_data.drop(columns=[target_protein]).equals(
                    test_data_sets[biopsy].drop(
                        columns=[target_protein])), "Data is not the same except for the target protein"

                adjusted_imputed_test_data[biopsy] = sub_test_data

            imp_tile_train_set: pd.DataFrame = create_tiles_for_dfs(adjusted_imputed_train_data.values(), tile_size,
                                                                    100)
            imp_tile_test_set: pd.DataFrame = create_tiles_for_dfs(adjusted_imputed_test_data.values(), tile_size, 100)

            assert target_protein in imp_tile_train_set.columns, "Target protein is not in the train data"
            assert target_protein in imp_tile_test_set.columns, "Target protein is not in the test data"

            assert imp_tile_train_set.shape[1] == imp_tile_test_set.shape[
                1], f"Train and test data shape is not similar, {len(imp_tile_train_set.columns)} != {len(imp_tile_test_set.columns)}"
            assert imp_tile_train_set.columns.equals(
                og_tile_train_set.columns), "Imputed data and original data columns are not the same"
            assert imp_tile_train_set.columns.equals(
                imp_tile_test_set.columns), "Imputed data and test data columns are not the same"

            imp_tile_train_sets[i] = imp_tile_train_set
            imp_tile_test_sets[i] = imp_tile_test_set

            rem_train_sets = {}
            for biopsy in train_data_sets.keys():
                rem_train_sets[biopsy] = train_data_sets[biopsy].copy()
                rem_train_sets[biopsy].drop(columns=[target_protein], inplace=True)

            rem_test_sets = {}
            for biopsy in test_data_sets.keys():
                rem_test_sets[biopsy] = test_data_sets[biopsy].copy()
                rem_test_sets[biopsy].drop(columns=[target_protein], inplace=True)

            rem_tile_train_set: pd.DataFrame = create_tiles_for_dfs(rem_train_sets.values(),
                                                                    removed_protein=target_protein, tile_size=tile_size,
                                                                    amount_of_tiles=100)
            rem_tile_test_set: pd.DataFrame = create_tiles_for_dfs(rem_test_sets.values(),
                                                                   removed_protein=target_protein, tile_size=tile_size,
                                                                   amount_of_tiles=100)

            # check that x_train removed does not include the target proteins
            assert target_protein not in rem_tile_train_set.columns, "Target protein is still in the data"
            assert target_protein not in rem_tile_test_set.columns, "Target protein is still in the data"

            assert rem_tile_train_set.shape[1] == rem_tile_test_set.shape[
                1], f"Train and test data shape is not similar, {len(rem_tile_train_set.columns)} != {len(rem_tile_test_set.columns)}"
            assert rem_tile_train_set.shape[1] != og_tile_train_set.shape[
                1], "Removed data and train data shape is not different"
            assert rem_tile_test_set.shape[1] != og_tile_test_set.shape[
                1], "Removed data and test data shape is not different"

            rem_tile_train_sets[i] = rem_tile_train_set
            rem_tile_test_sets[i] = rem_tile_test_set

        print("Running experiments...")
        for i in range(30):
            og_tile_train_set = og_tile_train_sets[i]
            og_tile_test_set = og_tile_test_sets[i]

            imp_tile_train_set = imp_tile_train_sets[i]
            imp_tile_test_set = imp_tile_test_sets[i]

            rem_tile_train_set = rem_tile_train_sets[i]
            rem_tile_test_set = rem_tile_test_sets[i]

            # run experiments
            og_experiment = ClassificationExperiment()
            og_experiment.setup(data=og_tile_train_set, target="Treatment",
                                index=True, normalize=True, normalize_method="minmax", verbose=False, fold=10,
                                fold_shuffle=True)
            og_classifier = og_experiment.create_model("lightgbm", verbose=False)
            og_best = og_experiment.compare_models([og_classifier], verbose=False)
            og_predictions = og_experiment.predict_model(og_best, data=og_tile_test_set, verbose=False)

            og_results = og_experiment.pull()
            # pull f1 score from the model
            og_score = og_results["Accuracy"].values[0]
            print(f"Score for protein {target_protein} using original data and bootstrap sample {i}: {og_score}")

            imp_experiment = ClassificationExperiment()
            imp_experiment.setup(data=imp_tile_train_set, target="Treatment", session_id=42, index=True,
                                 normalize=True, normalize_method="minmax", verbose=False, fold=10, fold_shuffle=True)
            imp_classifier = imp_experiment.create_model("lightgbm", verbose=False)
            imp_best = imp_experiment.compare_models([imp_classifier], verbose=False)

            imp_predictions = imp_experiment.predict_model(imp_best, data=imp_tile_test_set, verbose=False)

            imp_results = imp_experiment.pull()
            # pull the score from the model
            imp_score = imp_results["Accuracy"].values[0]

            print(f"Score for protein {target_protein} using imputed data and bootstrap sample {i}: {imp_score}")

            rem_experiment = ClassificationExperiment()
            rem_experiment.setup(data=rem_tile_train_set, target="Treatment", session_id=42,
                                 index=True, normalize=True, normalize_method="minmax", verbose=False, fold=10,
                                 fold_shuffle=True)
            rem_classifier = rem_experiment.create_model("lightgbm", verbose=False)
            rem_best = rem_experiment.compare_models([rem_classifier], verbose=False)

            rem_predictions = rem_experiment.predict_model(rem_best, data=rem_tile_test_set, verbose=False)

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
