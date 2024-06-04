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

save_path: Path = Path("results", "classifier", "informative_tiles")


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


def get_tile_coordinates(df, tile_size, amount_of_tiles, x_col='X_centroid',
                         y_col='Y_centroid'):
    x_min, x_max = df[x_col].min(), df[x_col].max() - tile_size
    y_min, y_max = df[y_col].min(), df[y_col].max() - tile_size

    # Generate more grid points for higher density
    grid_size = int(np.sqrt(amount_of_tiles) * 10)  # Further increase grid density
    x_points = np.linspace(x_min, x_max, num=grid_size)
    y_points = np.linspace(y_min, y_max, num=grid_size)

    tile_cells = []

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
                tile_df = tile_df[[x_col, y_col]]
                # add x_start, x_end, y_start, y_end to the tile_df
                tile_df["x_start"] = x_start_jittered
                tile_df["x_end"] = x_end
                tile_df["y_start"] = y_start_jittered
                tile_df["y_end"] = y_end
                tile_cells.append(tile_df)
    print(f"Found {len(tile_cells)} tiles")

    return tile_cells


def calculate_tile_features(tiles: [], data: pd.DataFrame) -> pd.DataFrame:
    tile_features = []

    for tile in tiles:
        # select the data based on the index
        tile_data = data.loc[tile.index]

        tile_mean = pd.DataFrame(tile_data.mean(numeric_only=True)).T
        tile_mean["cell_count"] = len(tile_data)

        # for col in SHARED_MARKERS:
        #    if col != "Treatment":
        #        tile_mean[f"{col}_mean"] = tile_data[col].mean()
        #        tile_mean[f"{col}_std"] = tile_data[col].std()

        if "Patient Id" in tile_mean.columns:
            tile_mean.drop("Patient Id", axis=1, inplace=True)

        tile_mean["Treatment"] = tile_data["Treatment"].values[0]
        # add x and y start and end
        tile_mean["x_start"] = tile["x_start"].values[0]
        tile_mean["x_end"] = tile["x_end"].values[0]
        tile_mean["y_start"] = tile["y_start"].values[0]
        tile_mean["y_end"] = tile["y_end"].values[0]

        assert "Patient Id" not in tile_mean.columns, "Patient Id column is in the dataset"
        assert "cell_type" not in tile_mean.columns, "Cell type column is in the dataset"
        assert "Treatment" in tile_mean.columns, "Treatment column is missing"

        tile_features.append(tile_mean)

    tile_features = pd.concat(tile_features, ignore_index=True)

    return tile_features


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


if __name__ == '__main__':

    exp = ClassificationExperiment()

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", "-p", type=str, required=True, help="Patient data to use")
    parser.add_argument("--tile_size", "-ts", type=int, default=350, help="Tile size to use")
    parser.add_argument("--iteration", "-i", type=int, default=1, help="Number of iterations to run the experiment")
    args = parser.parse_args()

    radius: int = 0
    patient: str = args.patient
    mode: str = "exp"
    tile_size: int = args.tile_size
    iterations: int = args.iteration

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

    for i in range(iterations):
        iteration_save_folder = Path(save_path, f"experiment_{i + 1}")
        if not iteration_save_folder.exists():
            iteration_save_folder.mkdir(parents=True)
        print(f"Iteration {i + 1}/{iterations}")

        # create tile coordinates
        print("Creating tiles for data...")
        # for each biopsy create tiles
        tile_train_sets = {}
        tile_test_sets = {}
        for biopsy in loaded_train_data_sets.keys():
            tile_train_sets[biopsy] = get_tile_coordinates(loaded_train_data_sets[biopsy], tile_size, 100)

        for biopsy in loaded_test_data_sets.keys():
            tile_test_sets[biopsy] = get_tile_coordinates(loaded_test_data_sets[biopsy], tile_size, 100)

        og_tile_train_sets = {}
        og_tile_test_sets = {}

        print("Calculating tile features...")
        for biopsy in loaded_train_data_sets.keys():
            og_tile_train_sets[biopsy] = calculate_tile_features(tiles=tile_train_sets[biopsy],
                                                                 data=loaded_train_data_sets[biopsy])

        for biopsy in loaded_test_data_sets.keys():
            og_tile_test_sets[biopsy] = calculate_tile_features(tiles=tile_test_sets[biopsy],
                                                                data=loaded_test_data_sets[biopsy])

        # save the tile sets for each biopsy
        for biopsy in og_tile_train_sets.keys():
            data_folder = Path(iteration_save_folder, "data")
            if not data_folder.exists():
                data_folder.mkdir(parents=True)

            og_tile_train_sets[biopsy].to_csv(Path(data_folder, f"{biopsy}_original_train_set.csv"),
                                              index=False)

        for biopsy in og_tile_test_sets.keys():
            data_folder = Path(iteration_save_folder, "data")
            if not data_folder.exists():
                data_folder.mkdir(parents=True)

            og_tile_test_sets[biopsy].to_csv(Path(data_folder, f"{biopsy}_original_test_set.csv"),
                                             index=False)

        for target_protein in SHARED_MARKERS:
            if target_protein == "Treatment":
                continue
            print(f"Running protein {target_protein}...")

            imp_train_sets = {}
            imp_test_sets = {}

            # replace the target protein with the imputed proteins value by train patients
            for biopsy in loaded_train_data_sets.keys():
                train_data = loaded_train_data_sets[biopsy].copy()
                imp_train_data = loaded_imputed_data[biopsy].copy()
                # replace target protein with imputed data using the indexes of the subset
                train_data.loc[:, target_protein] = imp_train_data.loc[:, target_protein]

                #  assert that target protein is not in the original data
                assert not train_data[target_protein].equals(
                    loaded_train_data_sets[biopsy][
                        target_protein]), "Target protein train data is equal to original data"

                # assert that except of the target protein the data is the same
                assert train_data.drop(columns=[target_protein]).equals(
                    loaded_train_data_sets[biopsy].drop(
                        columns=[target_protein])), "Data is not the same except for the target protein"

                imp_train_sets[biopsy] = train_data

            for biopsy in loaded_test_data_sets.keys():
                test_data = loaded_test_data_sets[biopsy].copy()
                imp_test_data = loaded_imputed_data[biopsy].copy()
                test_data.loc[:, target_protein] = imp_test_data.loc[:, target_protein]

                # assert that target proteins of subtest data is different compared to test_data_set
                assert not test_data[target_protein].equals(
                    loaded_test_data_sets[biopsy][target_protein]), "Target protein test data is equal to original data"

                # assert that except of the target protein the data is the same
                assert test_data.drop(columns=[target_protein]).equals(
                    loaded_test_data_sets[biopsy].drop(
                        columns=[target_protein])), "Data is not the same except for the target protein"

                imp_test_sets[biopsy] = test_data

            imp_tile_train_sets = {}
            imp_tile_test_sets = {}

            for biopsy in imp_train_sets.keys():
                imp_tile_train_sets[biopsy] = calculate_tile_features(tiles=tile_train_sets[biopsy],
                                                                      data=imp_train_sets[biopsy])

            for biopsy in imp_test_sets.keys():
                imp_tile_test_sets[biopsy] = calculate_tile_features(tiles=tile_test_sets[biopsy],
                                                                     data=imp_test_sets[biopsy])

            og_tile_train_set = pd.concat(og_tile_train_sets.values())
            og_tile_test_set = pd.concat(og_tile_test_sets.values())

            imp_tile_train_set = pd.concat(imp_tile_train_sets.values())
            imp_tile_test_set = pd.concat(imp_tile_test_sets.values())

            # reset the index
            og_tile_train_set.reset_index(drop=True, inplace=True)
            og_tile_test_set.reset_index(drop=True, inplace=True)
            imp_tile_train_set.reset_index(drop=True, inplace=True)
            imp_tile_test_set.reset_index(drop=True, inplace=True)

            imp_tile_test_spatial_data = imp_tile_test_set[["x_start", "x_end", "y_start", "y_end"]]
            og_tile_test_spatial_data = og_tile_test_set[["x_start", "x_end", "y_start", "y_end"]]

            assert imp_tile_test_spatial_data.equals(og_tile_test_spatial_data), "Spatial data is not the same"

            test_set_spatial_data = imp_tile_test_spatial_data.copy()

            # remove from the data the spatial information
            imp_tile_train_set.drop(columns=["x_start", "x_end", "y_start", "y_end"], inplace=True)
            imp_tile_test_set.drop(columns=["x_start", "x_end", "y_start", "y_end"], inplace=True)
            og_tile_train_set.drop(columns=["x_start", "x_end", "y_start", "y_end"], inplace=True)
            og_tile_test_set.drop(columns=["x_start", "x_end", "y_start", "y_end"], inplace=True)

            # run experiments
            og_experiment = ClassificationExperiment()
            og_experiment.setup(data=og_tile_train_set, target="Treatment",
                                index=True, normalize=True, normalize_method="minmax", verbose=False, fold=10,
                                fold_shuffle=True)
            og_classifier = og_experiment.create_model("lightgbm", verbose=False)
            og_best = og_experiment.compare_models([og_classifier], verbose=False)
            og_predictions = og_experiment.predict_model(og_best, data=og_tile_test_set, verbose=False)

            imp_experiment = ClassificationExperiment()
            imp_experiment.setup(data=imp_tile_train_set, target="Treatment", session_id=42, index=True,
                                 normalize=True, normalize_method="minmax", verbose=False, fold=10, fold_shuffle=True)
            imp_classifier = imp_experiment.create_model("lightgbm", verbose=False)
            imp_best = imp_experiment.compare_models([imp_classifier], verbose=False)

            imp_predictions = imp_experiment.predict_model(imp_best, data=imp_tile_test_set,
                                                           verbose=False)

            imp_results = imp_experiment.pull()
            # pull the score from the model
            imp_score = imp_results["Accuracy"].values[0]

            imp_predictions = imp_predictions[["Treatment", "prediction_label", "prediction_score", "cell_count"]]
            og_predictions = og_predictions[["Treatment", "prediction_label", "prediction_score", "cell_count"]]

            # combine the predictions, the truth treatment with the spatial coordinates of each tile based on the sets
            imp_predictions["x_start"] = test_set_spatial_data["x_start"]
            imp_predictions["x_end"] = test_set_spatial_data["x_end"]
            imp_predictions["y_start"] = test_set_spatial_data["y_start"]
            imp_predictions["y_end"] = test_set_spatial_data["y_end"]

            og_predictions["x_start"] = test_set_spatial_data["x_start"]
            og_predictions["x_end"] = test_set_spatial_data["x_end"]
            og_predictions["y_start"] = test_set_spatial_data["y_start"]
            og_predictions["y_end"] = test_set_spatial_data["y_end"]

            prediction_folder = Path(iteration_save_folder, "predictions")
            if not prediction_folder.exists():
                prediction_folder.mkdir(parents=True)

            # save the predictions
            imp_predictions.to_csv(Path(prediction_folder, f"{target_protein}_imputed_predictions.csv"),
                                   index=False)
            og_predictions.to_csv(Path(prediction_folder, f"{target_protein}_original_predictions.csv"),
                                  index=False)
