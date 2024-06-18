from pathlib import Path
import pandas as pd
import sys
import numpy as np

SHARED_MARKERS = ["pRB", "CD45", "CK19", "Ki67", "aSMA", "Ecad", "PR", "CK14", "HER2", "AR", "CK17", "p21", "Vimentin",
                  "pERK", "EGFR", "ER"]
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]


def find_matching_tiles(df):
    # Group by coordinates
    grouped = df.groupby(['x_start', 'x_end', 'y_start', 'y_end'])

    # Calculate the threshold for 90% of markers
    total_markers = len(df['Marker'].unique())
    threshold = 0.9 * total_markers

    # Filter groups that have at least 90% of all markers
    matching_groups = []
    for name, group in grouped:
        if len(group['Marker'].unique()) >= threshold:
            matching_groups.append(group)

    if not matching_groups:
        return pd.DataFrame()
    return pd.concat(matching_groups)


if __name__ == '__main__':

    for patient in PATIENTS:
        print(f"Processing patient: {patient}...")

        try:
            # Load the data
            pre_tx_path = Path("data", "bxs", f"{patient}_1.csv")
            post_tx_path = Path("data", "bxs", f"{patient}_2.csv")
            pre_tx = pd.read_csv(pre_tx_path)
            post_tx = pd.read_csv(post_tx_path)
        except:
            print(f"Base data for {patient} does not exist. Aborting!")
            continue

        imputed_correct_tiles = []
        original_correct_tiles = []
        for marker in SHARED_MARKERS:
            try:
                load_path = Path("results", "classifier", "informative_tiles", "exp", patient, "0", "experiment_1",
                                 "predictions")
                imputed_tiles = pd.read_csv(Path(load_path, f"{marker}_imputed_predictions.csv"))
                imputed_tiles["Marker"] = marker

                original_tiles = pd.read_csv(Path(load_path, f"{marker}_original_predictions.csv"))
                original_tiles["Marker"] = marker

                # correctly predicted tiles
                imputed_correct_tiles.append(
                    imputed_tiles[imputed_tiles["prediction_label"] == imputed_tiles["Treatment"]])
                original_correct_tiles.append(
                    original_tiles[original_tiles["prediction_label"] == original_tiles["Treatment"]])
            except:
                print(f"Data for {marker} for patient {patient} does not exist. Please run the downstream task first.")
                sys.exit(1)

        imputed_correct_tiles = pd.concat(imputed_correct_tiles)
        original_correct_tiles = pd.concat(original_correct_tiles)

        # Find tiles with matching coordinates across all markers
        imputed_matching_tiles = find_matching_tiles(imputed_correct_tiles)
        original_matching_tiles = find_matching_tiles(original_correct_tiles)

        imputed_pre_matching_tiles = imputed_matching_tiles[imputed_matching_tiles["Treatment"] == "PRE"]
        imputed_post_matching_tiles = imputed_matching_tiles[imputed_matching_tiles["Treatment"] == "ON"]

        original_pre_matching_tiles = original_matching_tiles[original_matching_tiles["Treatment"] == "PRE"]
        original_post_matching_tiles = original_matching_tiles[original_matching_tiles["Treatment"] == "ON"]

        unique_imputed_tiles = imputed_matching_tiles.drop_duplicates(subset=['x_start', 'x_end', 'y_start', 'y_end'])
        unique_original_tiles = original_matching_tiles.drop_duplicates(subset=['x_start', 'x_end', 'y_start', 'y_end'])

        print("Selecting cells...")

        # Extract columns for pre-matching
        x_start_pre = original_pre_matching_tiles['x_start'].values
        x_end_pre = original_pre_matching_tiles['x_end'].values
        y_start_pre = original_pre_matching_tiles['y_start'].values
        y_end_pre = original_pre_matching_tiles['y_end'].values

        # Create boolean masks for pre-matching
        mask_pre = (
                (pre_tx['X_centroid'].values[:, None] >= x_start_pre) &
                (pre_tx['X_centroid'].values[:, None] < x_end_pre) &
                (pre_tx['Y_centroid'].values[:, None] >= y_start_pre) &
                (pre_tx['Y_centroid'].values[:, None] < y_end_pre)
        )

        # Apply mask and concatenate results for pre-matching
        original_pre_matching_cells = pre_tx[np.any(mask_pre, axis=1)]

        # Extract columns for post-matching
        x_start_post = original_post_matching_tiles['x_start'].values
        x_end_post = original_post_matching_tiles['x_end'].values
        y_start_post = original_post_matching_tiles['y_start'].values
        y_end_post = original_post_matching_tiles['y_end'].values

        # Create boolean masks for post-matching
        mask_post = (
                (post_tx['X_centroid'].values[:, None] >= x_start_post) &
                (post_tx['X_centroid'].values[:, None] < x_end_post) &
                (post_tx['Y_centroid'].values[:, None] >= y_start_post) &
                (post_tx['Y_centroid'].values[:, None] < y_end_post)
        )

        # Apply mask and concatenate results for post-matching
        original_post_matching_cells = post_tx[np.any(mask_post, axis=1)]

        print("Dropping duplicates...")
        # only keep unique cells
        original_pre_matching_cells.drop_duplicates(subset=['X_centroid', 'Y_centroid'], inplace=True)
        original_post_matching_cells.drop_duplicates(subset=['X_centroid', 'Y_centroid'], inplace=True)

        print(original_post_matching_cells)
        print(original_pre_matching_cells)



        # check if any of the SHARED MARKERS is hihgly expressed in pre matching cells and post matching cells
        pre_highly_expressed_markers = []
        on_highly_expressed_markers = []
        for marker in SHARED_MARKERS:
            if np.mean(original_pre_matching_cells[marker]) > 0.5:
                pre_highly_expressed_markers.append(marker)
            if np.mean(original_post_matching_cells[marker]) > 0.5:
                on_highly_expressed_markers.append(marker)



        print(f"Patient: {patient}")
        print(f"Pre highly expressed markers: {pre_highly_expressed_markers}")
        print(f"Post highly expressed markers: {on_highly_expressed_markers}")
        input()

