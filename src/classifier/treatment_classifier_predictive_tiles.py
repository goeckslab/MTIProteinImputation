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

save_path: Path = Path("results", "classifier", "predictive_tissue")


def load_og_train_data(patient: str):
    train_data_sets = {}
    load_path = Path("results", "classifier", "same_tiles", "exp", patient, "0", "experiment_1", "data")
    for file in os.listdir(load_path):
        file_name = "_".join(Path(file).stem.split("_")[:3])
        if file.endswith(".csv") and 'original_train_set' in file:
            print("Loading train file: " + file)
            data = pd.read_csv(Path(load_path, file))

            assert "Treatment" in data.columns, f"Treatment column is missing for dataframe of patient {file}"
            # merge SHARED MARKER with X and Y centroid
            shared_columns = SHARED_MARKERS + ["X_centroid", "Y_centroid"] + ["x_start", "x_end", "y_start", "y_end"]
            data = data[shared_columns]

            train_data_sets[file_name] = data

    assert len(train_data_sets) == 6, f"There should be 6 train datasets, loaded {len(train_data_sets)}"
    return train_data_sets


def load_og_test_data(patient: str):
    test_data_sets = {}
    load_path = Path("results", "classifier", "same_tiles", "exp", patient, "0", "experiment_1", "data")
    for file in os.listdir(load_path):
        file_name = "_".join(Path(file).stem.split("_")[:3])
        if file.endswith(".csv") and 'original_test_set' in file:
            print("Loading test file: " + file)
            data = pd.read_csv(Path(load_path, file))

            assert "Treatment" in data.columns, f"Treatment column is missing for dataframe of patient {file}"
            # merge SHARED MARKER with X and Y centroid
            shared_columns = SHARED_MARKERS + ["X_centroid", "Y_centroid"] + ["x_start", "x_end", "y_start", "y_end"]
            data = data[shared_columns]

            test_data_sets[file_name] = data

    assert len(test_data_sets) == 2, f"There should be 2 test datasets, loaded {len(test_data_sets)}"
    return test_data_sets


def load_imp_train_data(patient: str, protein: str):
    train_data_sets = {}
    load_path = Path("results", "classifier", "same_tiles", "exp", patient, "0", "experiment_1", "data")
    for file in os.listdir(load_path):
        file_name = "_".join(Path(file).stem.split("_")[:3])
        if file.endswith(".csv") and f'{protein}_imputed_train_set' in file:
            print("Loading train file: " + file)
            data = pd.read_csv(Path(load_path, file))

            assert "Treatment" in data.columns, f"Treatment column is missing for dataframe of patient {file}"
            # merge SHARED MARKER with X and Y centroid
            shared_columns = SHARED_MARKERS + ["X_centroid", "Y_centroid"] + ["x_start", "x_end", "y_start", "y_end"]
            data = data[shared_columns]

            train_data_sets[file_name] = data

    assert len(train_data_sets) == 6, f"There should be 6 train datasets, loaded {len(train_data_sets)}"
    return train_data_sets


def load_imp_test_data(patient: str, protein: str):
    test_data_sets = {}
    load_path = Path("results", "classifier", "same_tiles", "exp", patient, "0", "experiment_1", "data")
    for file in os.listdir(load_path):
        file_name = "_".join(Path(file).stem.split("_")[:3])
        if file.endswith(".csv") and f'{protein}_imputed_test_set' in file:
            print("Loading test file: " + file)
            data = pd.read_csv(Path(load_path, file))

            assert "Treatment" in data.columns, f"Treatment column is missing for dataframe of patient {file}"
            # merge SHARED MARKER with X and Y centroid
            shared_columns = SHARED_MARKERS + ["X_centroid", "Y_centroid"] + ["x_start", "x_end", "y_start", "y_end"]
            data = data[shared_columns]

            test_data_sets[file_name] = data

    assert len(test_data_sets) == 2, f"There should be 2 test datasets, loaded {len(test_data_sets)}"
    return test_data_sets


def load_predictive_tiles():
    tmp_og_predictive_tiles = {}
    tmp_imp_predictive_tiles = {}
    for tmp_patient in PATIENTS:
        load_path = Path("results", "predictive_tissue", tmp_patient)
        original_tiles = pd.read_csv(Path(load_path, "original_matching_tiles.csv"))
        imputed_tiles = pd.read_csv(Path(load_path, "imputed_matching_tiles.csv"))
        tmp_og_predictive_tiles[tmp_patient] = original_tiles
        tmp_imp_predictive_tiles[tmp_patient] = imputed_tiles

    return tmp_og_predictive_tiles, tmp_imp_predictive_tiles


def extract_predictive_tiles(data: {}, tiles: {}, patient: str):
    for tmp_patient in tiles.keys():
        if tmp_patient == patient:
            continue
        print("TMP")
        print(tmp_patient)
        tmp_patient_tiles = tiles[tmp_patient]
        pre_tiles = tmp_patient_tiles[tmp_patient_tiles["Treatment"] == "PRE"]
        post_tiles = tmp_patient_tiles[tmp_patient_tiles["Treatment"] == "ON"]
        print(pre_tiles)
        #print(post_tiles)
        print("DATA")
        pre_patient_data = data[f"{tmp_patient}_1"]
        post_patient_data = data[f"{tmp_patient}_2"]
        print(pre_patient_data)
        #print(post_patient_data)
        input()

        # rename X_start, X_end, Y_start, Y_end to x_start, x_end, y_start, y_end
        pre_tiles.rename(columns={"X_start": "x_start", "X_end": "x_end", "Y_start": "y_start", "Y_end": "y_end"},
                         inplace=True)
        post_tiles.rename(columns={"X_start": "x_start", "X_end": "x_end", "Y_start": "y_start", "Y_end": "y_end"},
                          inplace=True)

        pre_patient_data = pd.merge(pre_patient_data, pre_tiles, on=["x_start", "x_end", "y_start", "y_end"])
        post_patient_data = pd.merge(post_patient_data, post_tiles, on=["x_start", "x_end", "y_start", "y_end"])

        print(pre_patient_data)
        #print(post_patient_data)

        input()
    input()

    return


if __name__ == '__main__':

    exp = ClassificationExperiment()

    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', "-r", type=int, default=0, choices=[0, 15, 30, 60, 90, 120])
    parser.add_argument("--patient", "-p", type=str, required=True, help="Patient data to use")
    parser.add_argument("--tile_size", "-ts", type=int, default=350, help="Tile size to use")
    parser.add_argument("--iteration", "-i", type=int, default=30, help="Number of iterations to run the experiment")
    args = parser.parse_args()

    radius: int = args.radius
    patient: str = args.patient
    mode: str = "exp"
    tile_size: int = args.tile_size
    iterations: int = args.iteration

    print(f"Using radius: {radius} µm")
    print(f"Using patient: {patient}")
    print(f"Using mode: {mode}")

    save_path = Path(save_path, mode, patient, str(radius))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    og_predictive_tiles, imp_predictive_tiles = load_predictive_tiles()

    scores = []
    for i in range(iterations):
        iteration_save_folder = Path(save_path, f"experiment_{i + 1}")
        if not iteration_save_folder.exists():
            iteration_save_folder.mkdir(parents=True)
        print(f"Iteration {i + 1}/{iterations}")

        og_train_data_sets = load_og_train_data(patient=patient)
        og_test_data_sets = load_og_test_data(patient=patient)

        og_train_data_sets = extract_predictive_tiles(data=og_train_data_sets,
                                                      tiles=og_predictive_tiles, patient=patient)

        og_tile_train_set = pd.concat(og_train_data_sets.values())
        og_tile_test_set = pd.concat(og_test_data_sets.values())

        for target_protein in SHARED_MARKERS:
            if target_protein == "Treatment":
                continue
            print(f"Running protein {target_protein}...")

            imp_train_data_sets = load_imp_train_data(patient=patient, protein=target_protein)
            imp_test_data_sets = load_imp_test_data(patient=patient, protein=target_protein)

            imp_tile_train_set = pd.concat(imp_train_data_sets.values())
            imp_tile_test_set = pd.concat(imp_test_data_sets.values())

            # reset the index
            og_tile_train_set.reset_index(drop=True, inplace=True)
            og_tile_test_set.reset_index(drop=True, inplace=True)
            imp_tile_train_set.reset_index(drop=True, inplace=True)
            imp_tile_test_set.reset_index(drop=True, inplace=True)

            imp_tile_test_spatial_data = imp_tile_test_set[["X_centroid", "Y_centroid"]]
            og_tile_test_spatial_data = og_tile_test_set[["X_centroid", "Y_centroid"]]

            # remove from the data the spatial information
            imp_tile_train_set.drop(columns=["X_centroid", "Y_centroid"], inplace=True)
            imp_tile_test_set.drop(columns=["X_centroid", "Y_centroid"], inplace=True)
            og_tile_train_set.drop(columns=["X_centroid", "Y_centroid"], inplace=True)
            og_tile_test_set.drop(columns=["X_centroid", "Y_centroid"], inplace=True)

            print("Running experiment...")

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

            imp_predictions = imp_experiment.predict_model(imp_best, data=imp_tile_test_set,
                                                           verbose=False)

            imp_results = imp_experiment.pull()
            # pull the score from the model
            imp_score = imp_results["Accuracy"].values[0]

            print(f"Score for protein {target_protein} using imputed data: {imp_score}")

            scores.append({"Protein": target_protein, "Imputed Score": imp_score, "Original Score": og_score})
            # print current mean of scores for protein
            print(
                f"Score for protein {target_protein}")
            print(pd.DataFrame(scores).groupby('Protein').mean())

        scores = pd.DataFrame(scores)
        # calculate mean of proteins
        mean_scores = scores.groupby("Protein").mean().reset_index()
        mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
        scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
