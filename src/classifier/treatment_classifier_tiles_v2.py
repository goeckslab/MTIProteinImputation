from pycaret.classification import ClassificationExperiment
from pathlib import Path
import pandas as pd
import os, sys, argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER', "Treatment"]
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier", "pycaret_tiles_spatial_information")

iterations = 30


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
        pre_treatment_path = Path("results", "imputed_data", "ae", "single", mode, patient, f"{radius}",
                                  "pre_treatment.csv")
        on_treatment_path = Path("results", "imputed_data", "ae", "single", mode, patient, f"{radius}",
                                 "on_treatment.csv")

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


def create_tiles_for_df(df, tile_size, amount_of_tiles, removed_protein: str = '', x_col='X_centroid',
                        y_col='Y_centroid'):
    x_min, x_max = df[x_col].min(), df[x_col].max() - tile_size
    y_min, y_max = df[y_col].min(), df[y_col].max() - tile_size

    # Generate more grid points for higher density
    grid_size = int(np.sqrt(amount_of_tiles) * 2)  # Further increase grid density
    x_points = np.linspace(x_min, x_max, num=grid_size)
    y_points = np.linspace(y_min, y_max, num=grid_size)

    tile_data = []
    tile_spatial_data = []

    for x_start in x_points:
        for y_start in y_points:
            # Add jitter to cover more space
            x_jitter = np.random.uniform(-tile_size / 4, tile_size / 4)
            y_jitter = np.random.uniform(-tile_size / 4, tile_size / 4)
            x_start_jittered = np.clip(x_start + x_jitter, x_min, x_max)
            y_start_jittered = np.clip(y_start + y_jitter, y_min, y_max)

            x_end = x_start_jittered + tile_size
            y_end = y_start_jittered + tile_size

            mask = (
                    (df[x_col] >= x_start_jittered) & (df[x_col] < x_end) &
                    (df[y_col] >= y_start_jittered) & (df[y_col] < y_end)
            )
            tile_df = df[mask]

            if not tile_df.empty and len(tile_df) >= 50:
                features = pd.DataFrame(tile_df.mean(numeric_only=True)).T

                for col in SHARED_MARKERS:
                    if col != "Treatment" and col != removed_protein:
                        features[f"{col}_mean"] = tile_df[col].mean()
                        features[f"{col}_std"] = tile_df[col].std()

                features["cell_count"] = len(tile_df)
                features["Treatment"] = tile_df["Treatment"].iloc[
                    0] if "Treatment" in tile_df.columns else "Not Available"

                if "Patient Id" in features.columns:
                    features.drop("Patient Id", axis=1, inplace=True)

                assert "Patient Id" not in features.columns, "Patient Id column is in the dataset"
                assert "cell_type" not in features.columns, "Cell type column is in the dataset"
                assert "Treatment" in features.columns, "Treatment column is missing"

                tile_data.append(features)
                tile_spatial_data.append({
                    "x_start": x_start_jittered,
                    "x_end": x_end,
                    "y_start": y_start_jittered,
                    "y_end": y_end,
                })

    tiles_df = pd.concat(tile_data, ignore_index=True)
    tile_spatial_data = pd.DataFrame(tile_spatial_data)
    tiles_df.fillna(0, inplace=True)

    assert len(tiles_df) == len(tile_spatial_data), "Tile data and spatial data are not the same length"

    print(f"Found {len(tiles_df)} tiles")

    return tiles_df, tile_spatial_data


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
                         removed_protein: str = '') -> (pd.DataFrame, pd.DataFrame):
    # create tiles for original data
    tile_sets = []
    tile_spatial_sets = []
    for data_set in dataframes:
        tile_set, tile_spatial_set = create_tiles_for_df(data_set, tile_size, amount_of_tiles,
                                                         removed_protein=removed_protein)
        tile_sets.append(tile_set)
        tile_spatial_sets.append(tile_spatial_set)

    tile_set = pd.concat(tile_sets)
    tile_set.reset_index(drop=True, inplace=True)

    tile_spatial_set = pd.concat(tile_spatial_sets)
    tile_spatial_set.reset_index(drop=True, inplace=True)

    return tile_set, tile_spatial_set


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

        print("Preparing tiles for data...")
        train_data_sets = {}
        test_data_sets = {}

        # copy each dataset in the dictionary
        for key in loaded_train_data_sets.keys():
            train_data_sets[key] = loaded_train_data_sets[key].copy()

        for key in loaded_test_data_sets.keys():
            test_data_sets[key] = loaded_test_data_sets[key].copy()

        og_tile_train_set, _ = create_tiles_for_dfs(train_data_sets.values(), tile_size, 100)

        # treatment should be both pre and on
        assert len(og_tile_train_set["Treatment"].unique()) == 2, "Treatment should be both pre and on"

        og_tile_test_set, og_tile_spatial_data = create_tiles_for_dfs(test_data_sets.values(), tile_size, 100)
        assert len(og_tile_test_set["Treatment"].unique()) == 2, "Treatment should be both pre and on"

        # assert that target protein is in train and test data
        assert target_protein in og_tile_train_set.columns, "Target protein is not in the train data"
        assert target_protein in og_tile_test_set.columns, "Target protein is not in the test data"

        adjusted_imputed_train_data = {}
        # create new imp dataset from original data and replace
        # the target protein with the imputed proteins value by train patients
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

        imp_tile_train_set, _ = create_tiles_for_dfs(adjusted_imputed_train_data.values(), tile_size,
                                                     100)
        imp_tile_test_set, imp_tile_spatial_data = create_tiles_for_dfs(adjusted_imputed_test_data.values(), tile_size,
                                                                        100)

        # assert tile test and tile spatial data same length
        assert len(imp_tile_test_set) == len(
            imp_tile_spatial_data), "Tile test and tile spatial data are not the same length"

        assert target_protein in imp_tile_train_set.columns, "Target protein is not in the train data"
        assert target_protein in imp_tile_test_set.columns, "Target protein is not in the test data"

        assert imp_tile_train_set.shape[1] == imp_tile_test_set.shape[
            1], f"Train and test data shape is not similar, {len(imp_tile_train_set.columns)} != {len(imp_tile_test_set.columns)}"
        assert imp_tile_train_set.columns.equals(
            og_tile_train_set.columns), "Imputed data and original data columns are not the same"
        assert imp_tile_train_set.columns.equals(
            imp_tile_test_set.columns), "Imputed data and test data columns are not the same"

        print("Running experiments...")

        # run experiments
        og_experiment = ClassificationExperiment()
        og_experiment.setup(data=og_tile_train_set, target="Treatment",
                            index=True, normalize=True, normalize_method="minmax", verbose=False, fold=10,
                            fold_shuffle=True)
        og_classifier = og_experiment.create_model("lightgbm", verbose=False)
        og_best = og_experiment.compare_models([og_classifier], verbose=False)
        og_predictions = og_experiment.predict_model(og_best, data=og_tile_test_set, verbose=False)
        print(og_predictions.groupby("Treatment").size())

        og_results = og_experiment.pull()
        # pull f1 score from the model
        og_score = og_results["Accuracy"].values[0]
        print(f"Score for protein {target_protein} using original data: {og_score}")

        imp_experiment = ClassificationExperiment()
        imp_experiment.setup(data=imp_tile_train_set, target="Treatment", session_id=42, index=True,
                             normalize=True, normalize_method="minmax", verbose=False, fold=10, fold_shuffle=True)
        imp_classifier = imp_experiment.create_model("lightgbm", verbose=False)
        imp_best = imp_experiment.compare_models([imp_classifier], verbose=False)

        imp_predictions = imp_experiment.predict_model(imp_best, data=imp_tile_test_set, verbose=False)

        imp_results = imp_experiment.pull()
        # pull the score from the model
        imp_score = imp_results["Accuracy"].values[0]

        print(f"Score for protein {target_protein} using imputed data: {imp_score}")

        scores.append({"Protein": target_protein, "Imputed Score": imp_score, "Original Score": og_score})
        # print current mean of scores for protein
        print(
            f"Score for protein {target_protein}")
        print(pd.DataFrame(scores).groupby('Protein').mean())

        imp_predictions = imp_predictions[["Treatment", "prediction_label", "prediction_score", "cell_count"]]
        og_predictions = og_predictions[["Treatment", "prediction_label", "prediction_score", "cell_count"]]

        # combine the predictions, the truth treatment and the spatial information for the imputed data
        imp_predictions["X_start"] = imp_tile_spatial_data["x_start"]
        imp_predictions["X_end"] = imp_tile_spatial_data["x_end"]
        imp_predictions["Y_start"] = imp_tile_spatial_data["y_start"]
        imp_predictions["Y_end"] = imp_tile_spatial_data["y_end"]

        og_predictions["X_start"] = og_tile_spatial_data["x_start"]
        og_predictions["X_end"] = og_tile_spatial_data["x_end"]
        og_predictions["Y_start"] = og_tile_spatial_data["y_start"]
        og_predictions["Y_end"] = og_tile_spatial_data["y_end"]

        # save the predictions
        imp_predictions.to_csv(Path(save_path, f"{target_protein}_imputed_predictions.csv"), index=False)
        og_predictions.to_csv(Path(save_path, f"{target_protein}_original_predictions.csv"), index=False)

    scores = pd.DataFrame(scores)
    # calculate mean of proteins
    mean_scores = scores.groupby("Protein").mean().reset_index()
    mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
