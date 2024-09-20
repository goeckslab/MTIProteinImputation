from pycaret.classification import ClassificationExperiment
from pathlib import Path
import pandas as pd
import argparse
import numpy as np

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

save_path: Path = Path("results", "classifier", "downstream_classifier")


# Function to check if a point is within a tile
def is_within_tile(x, y, tile):
    return (tile['X_start'] <= x <= tile['X_end']) and (tile['Y_start'] <= y <= tile['Y_end'])


def load_predictive_tiles():
    tmp_og_predictive_tiles = {}
    tmp_removed_predictive_tiles = {}
    for tmp_patient in PATIENTS:
        load_path = Path("results", "predictive_tissue", tmp_patient)
        original_tiles = pd.read_csv(Path(load_path, "original_matching_tiles.csv"))
        removed_tiles = pd.read_csv(Path(load_path, "removed_matching_tiles.csv"))
        tmp_og_predictive_tiles[tmp_patient] = original_tiles
        tmp_removed_predictive_tiles[tmp_patient] = removed_tiles

    return tmp_og_predictive_tiles, tmp_removed_predictive_tiles


def extract_og_cells_for_biopsy(biopsy: str, tiles: pd.DataFrame, pre_treatment: bool):
    biopsy_path = Path("data", "bxs", f"{biopsy}.csv")
    cells = pd.read_csv(biopsy_path)
    # Convert the DataFrame columns to numpy arrays for faster operations
    x_start = tiles['x_start'].values
    x_end = tiles['x_end'].values
    y_start = tiles['y_start'].values
    y_end = tiles['y_end'].values

    x_centroid = cells['X_centroid'].values
    y_centroid = cells['Y_centroid'].values

    # Create arrays for comparison
    x_centroid = x_centroid[:, np.newaxis]
    y_centroid = y_centroid[:, np.newaxis]

    # Perform vectorized comparison
    in_x_range = (x_centroid >= x_start) & (x_centroid <= x_end)
    in_y_range = (y_centroid >= y_start) & (y_centroid <= y_end)
    in_range = in_x_range & in_y_range

    # Create a list to store the results
    results = []

    # For each tile, select the matching cells and store them in the results list
    for i in range(len(tiles)):
        selected_cells = cells[in_range[:, i]].copy()
        results.append(selected_cells)

    # Concatenate all results into a single DataFrame
    final_result = pd.concat(results, ignore_index=True)
    # drop all duplicates
    final_result.drop_duplicates(inplace=True)
    final_result = final_result[SHARED_MARKERS]

    if pre_treatment:
        final_result["Treatment"] = "PRE"
    else:
        final_result["Treatment"] = "ON"

    return final_result


def extract_imp_cells_for_biopsy(biopsy: str, tiles: pd.DataFrame, protein: str, pre_treatment: bool):
    biopsy_path = Path("data", "bxs", f"{biopsy}.csv")
    if pre_treatment:
        imp_treatment = pd.read_csv(
            Path("results", "imputed_data", "ae", "single", "exp", patient, "0", "pre_treatment.csv"))
    else:
        imp_treatment = pd.read_csv(
            Path("results", "imputed_data", "ae", "single", "exp", patient, "0", "on_treatment.csv"))

    cells = pd.read_csv(biopsy_path)

    original_expression = cells[protein]

    # replace the protein in cells with the data from imp treatment protein
    cells[protein] = imp_treatment[protein]

    # assert that the original expression is not the same as the imputed expression
    assert not np.array_equal(original_expression,
                              cells[protein]), f"Original expression is the same as imputed expression"
    # assert that original expression is different from cells protein
    assert not np.array_equal(original_expression, cells[protein]), f"Original expression is the same as cells protein"

    # Convert the DataFrame columns to numpy arrays for faster operations
    x_start = tiles['x_start'].values
    x_end = tiles['x_end'].values
    y_start = tiles['y_start'].values
    y_end = tiles['y_end'].values

    x_centroid = cells['X_centroid'].values
    y_centroid = cells['Y_centroid'].values

    # Create arrays for comparison
    x_centroid = x_centroid[:, np.newaxis]
    y_centroid = y_centroid[:, np.newaxis]

    # Perform vectorized comparison
    in_x_range = (x_centroid >= x_start) & (x_centroid <= x_end)
    in_y_range = (y_centroid >= y_start) & (y_centroid <= y_end)
    in_range = in_x_range & in_y_range

    # Create a list to store the results
    results = []

    # For each tile, select the matching cells and store them in the results list
    for i in range(len(tiles)):
        selected_cells = cells[in_range[:, i]].copy()
        results.append(selected_cells)

    # Concatenate all results into a single DataFrame
    final_result = pd.concat(results, ignore_index=True)
    # drop all duplicates
    final_result.drop_duplicates(inplace=True)
    final_result = final_result[SHARED_MARKERS]
    if pre_treatment:
        final_result["Treatment"] = "PRE"
    else:
        final_result["Treatment"] = "ON"

    return final_result


if __name__ == '__main__':

    exp = ClassificationExperiment()

    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', "-r", type=int, default=0, choices=[0, 15, 30, 60, 90, 120])
    parser.add_argument("--patient", "-p", type=str, required=True, help="Patient data to use")
    parser.add_argument("--iteration", "-i", type=int, default=30, help="Number of iterations to run the experiment")
    args = parser.parse_args()

    radius: int = args.radius
    patient: str = args.patient
    mode: str = "exp"
    iterations: int = args.iteration

    print(f"Using radius: {radius} µm")
    print(f"Using patient: {patient}")
    print(f"Using mode: {mode}")
    print(f"Running {iterations} iterations")

    save_path = Path(save_path, mode, patient, str(radius))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    _, removed_predictive_tiles = load_predictive_tiles()

    og_data = {}

    # for each patient in the tiles dictioniary, load the biopsy data and extract the cells
    for tmp_patient, tiles in removed_predictive_tiles.items():
        pre_tiles = tiles[tiles["Treatment"] == "PRE"]
        post_tiles = tiles[tiles["Treatment"] == "ON"]

        pre_biopsy = f"{tmp_patient}_1"
        post_biopsy = f"{tmp_patient}_2"

        pre_biopsy_data = extract_og_cells_for_biopsy(pre_biopsy, pre_tiles, pre_treatment=True)
        post_biopsy_data = extract_og_cells_for_biopsy(post_biopsy, post_tiles, pre_treatment=False)

        og_data[pre_biopsy] = pre_biopsy_data
        og_data[post_biopsy] = post_biopsy_data

    scores = []
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        for target_protein in SHARED_MARKERS:
            imp_data = {}
            # Load the imputed predictive tiles
            for tmp_patient, tiles in removed_predictive_tiles.items():
                pre_tiles = tiles[tiles["Treatment"] == "PRE"]
                post_tiles = tiles[tiles["Treatment"] == "ON"]

                pre_biopsy = f"{tmp_patient}_1"
                post_biopsy = f"{tmp_patient}_2"

                pre_biopsy_data = extract_imp_cells_for_biopsy(pre_biopsy, pre_tiles, protein=target_protein,
                                                               pre_treatment=True)
                post_biopsy_data = extract_imp_cells_for_biopsy(post_biopsy, post_tiles, protein=target_protein,
                                                                pre_treatment=False)

                imp_data[pre_biopsy] = pre_biopsy_data
                imp_data[post_biopsy] = post_biopsy_data

            # create dictionary comprehension to select all biopsies for the current patient
            imp_test_data = {biopsy: data for biopsy, data in imp_data.items() if patient in biopsy}
            og_test_data = {biopsy: data for biopsy, data in og_data.items() if patient in biopsy}
            removed_test_data = {biopsy: data.copy() for biopsy, data in og_data.items() if patient in biopsy}

            imp_train_data = {biopsy: data for biopsy, data in imp_data.items() if patient not in biopsy}
            og_train_data = {biopsy: data for biopsy, data in og_data.items() if patient not in biopsy}
            removed_train_data = {biopsy: data.copy() for biopsy, data in og_data.items() if patient not in biopsy}

            # create train sets
            imp_train_set = pd.concat(imp_train_data.values())
            og_train_set = pd.concat(og_train_data.values())
            removed_train_set = pd.concat(removed_train_data.values())

            # bootstrap the data and use only 80%
            imp_train_set = imp_train_set.sample(frac=0.8, random_state=42)
            og_train_set = og_train_set.sample(frac=0.8, random_state=42)
            removed_train_set = removed_train_set.sample(frac=0.8, random_state=42)

            # reset index
            imp_train_set.reset_index(drop=True, inplace=True)
            og_train_set.reset_index(drop=True, inplace=True)
            removed_train_set.reset_index(drop=True, inplace=True)

            # create test sets
            imp_test_set = pd.concat(imp_test_data.values())
            og_test_set = pd.concat(og_test_data.values())
            removed_test_set = pd.concat(removed_test_data.values())

            # bootstrap the data and use only 80%
            imp_test_set = imp_test_set.sample(frac=0.8, random_state=42)
            og_test_set = og_test_set.sample(frac=0.8, random_state=42)
            removed_test_set = removed_test_set.sample(frac=0.8, random_state=42)

            # reset index
            imp_test_set.reset_index(drop=True, inplace=True)
            og_test_set.reset_index(drop=True, inplace=True)
            removed_test_set.reset_index(drop=True, inplace=True)

            # remove target protein from removed data
            removed_train_set.drop(columns=[target_protein], inplace=True)
            removed_test_set.drop(columns=[target_protein], inplace=True)

            assert target_protein not in removed_train_set.columns, f"Target protein {target_protein} still in removed_train_set"
            assert target_protein not in removed_test_set.columns, f"Target protein {target_protein} still in removed_test_set"
            assert len(removed_train_set.columns) == len(
                og_train_set.columns) - 1, f"Length of removed_train_set and og_train_set should be -1 length, but got {len(removed_train_set)} and {len(og_train_set)}"
            assert len(removed_test_set.columns) == len(
                og_test_set.columns) - 1, f"Length of removed_test_set and og_test_set should be -1 length, but got {len(removed_test_set)} and {len(og_test_set)}"

            print(f"Running experiment for target: {target_protein}")

            # run experiments
            og_experiment = ClassificationExperiment()
            og_experiment.setup(data=og_train_set, target="Treatment",
                                index=True, normalize=True, normalize_method="minmax", verbose=False, fold=10,
                                fold_shuffle=True)
            og_classifier = og_experiment.create_model("lightgbm", verbose=False)
            og_predictions = og_experiment.predict_model(og_classifier, data=og_test_set, verbose=False)

            og_results = og_experiment.pull()
            # pull f1 score from the model
            og_score = og_results["Accuracy"].values[0]
            print(f"Score for protein {target_protein} using original data: {og_score}")

            # Run experiment with removed target protein
            rm_experiment = ClassificationExperiment()
            rm_experiment.setup(data=removed_train_set, target="Treatment", index=True,
                                normalize=True, normalize_method="minmax", verbose=False, fold=10, fold_shuffle=True)
            rm_classifier = rm_experiment.create_model("lightgbm", verbose=False)
            rm_predictions = rm_experiment.predict_model(rm_classifier, data=removed_test_set,
                                                         verbose=False)

            rm_results = rm_experiment.pull()
            # pull the score from the model
            rm_score = rm_results["Accuracy"].values[0]

            print(f"Score for protein {target_protein} using removed data: {rm_score}")

            imp_experiment = ClassificationExperiment()
            imp_experiment.setup(data=imp_train_set, target="Treatment", index=True,
                                 normalize=True, normalize_method="minmax", verbose=False, fold=10, fold_shuffle=True)
            imp_classifier = imp_experiment.create_model("lightgbm", verbose=False)
            imp_predictions = imp_experiment.predict_model(imp_classifier, data=imp_test_set,
                                                           verbose=False)

            imp_results = imp_experiment.pull()
            # pull the score from the model
            imp_score = imp_results["Accuracy"].values[0]

            print(f"Score for protein {target_protein} using imputed data: {imp_score}")

            scores.append({"Protein": target_protein, "Imputed Score": imp_score, "Original Score": og_score,
                           "Removed Score": rm_score})
            # print current mean of scores for protein
            print(
                f"Score for protein {target_protein}")
            print(pd.DataFrame(scores).groupby('Protein').mean())

    scores = pd.DataFrame(scores)
    # calculate mean of proteins
    mean_scores = scores.groupby("Protein").mean().reset_index()
    mean_scores.to_csv(Path(save_path, "mean_classifier_scores.csv"), index=False)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
