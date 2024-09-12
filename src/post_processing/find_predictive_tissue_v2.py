from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

SHARED_MARKERS = ["pRB", "CD45", "CK19", "Ki67", "aSMA", "Ecad", "PR", "CK14", "HER2", "AR", "CK17", "p21", "Vimentin",
                  "pERK", "EGFR", "ER"]
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

figure_save_path = Path("figures", "tiles")
data_save_path = Path("results", "predictive_tissue")


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

        patient_figure_save_path = Path(figure_save_path, patient)
        patient_data_save_path = Path(data_save_path, patient)
        if not patient_figure_save_path.exists():
            patient_figure_save_path.mkdir(parents=True)

        if not patient_data_save_path.exists():
            patient_data_save_path.mkdir(parents=True)

        try:
            # Load the data
            pre_tx_path = Path("data", "bxs", f"{patient}_1.csv")
            post_tx_path = Path("data", "bxs", f"{patient}_2.csv")
            pre_tx = pd.read_csv(pre_tx_path)
            post_tx = pd.read_csv(post_tx_path)
        except:
            print(f"Base data for {patient} does not exist. Aborting!")
            continue

        removed_correct_tiles = []
        original_correct_tiles = []
        for marker in SHARED_MARKERS:
            try:
                load_path = Path("results", "classifier", "informative_tiles", "exp", patient, "0", "experiment_1",
                                 "predictions")
                removed_tiles = pd.read_csv(Path(load_path, f"{marker}_removed_predictions.csv"))
                removed_tiles["Marker"] = marker

                original_tiles = pd.read_csv(Path(load_path, f"{marker}_original_predictions.csv"))
                original_tiles["Marker"] = marker

                # correctly predicted tiles
                removed_correct_tiles.append(
                    removed_tiles[removed_tiles["prediction_label"] == removed_tiles["Treatment"]])
                original_correct_tiles.append(
                    original_tiles[original_tiles["prediction_label"] == original_tiles["Treatment"]])
            except:
                print(f"Data for {marker} for patient {patient} does not exist. Please run the downstream task first.")
                sys.exit(1)

        removed_correct_tiles = pd.concat(removed_correct_tiles)
        original_correct_tiles = pd.concat(original_correct_tiles)

        # Find tiles with matching coordinates across all markers
        removed_matching_tiles = find_matching_tiles(removed_correct_tiles)
        original_matching_tiles = find_matching_tiles(original_correct_tiles)

        removed_pre_matching_tiles = removed_matching_tiles[removed_matching_tiles["Treatment"] == "PRE"]
        removed_post_matching_tiles = removed_matching_tiles[removed_matching_tiles["Treatment"] == "ON"]

        removed_post_matching_tiles.to_csv(Path(patient_data_save_path, "imputed_post_matching_tiles.csv"), index=False)
        removed_pre_matching_tiles.to_csv(Path(patient_data_save_path, "imputed_pre_matching_tiles.csv"), index=False)

        original_pre_matching_tiles = original_matching_tiles[original_matching_tiles["Treatment"] == "PRE"]
        original_post_matching_tiles = original_matching_tiles[original_matching_tiles["Treatment"] == "ON"]

        original_pre_matching_tiles.to_csv(Path(patient_data_save_path, "original_pre_matching_tiles.csv"), index=False)
        original_post_matching_tiles.to_csv(Path(patient_data_save_path, "original_post_matching_tiles.csv"),
                                            index=False)

        unique_removed_tiles = removed_matching_tiles.drop_duplicates(subset=['x_start', 'x_end', 'y_start', 'y_end'])
        unique_original_tiles = original_matching_tiles.drop_duplicates(subset=['x_start', 'x_end', 'y_start', 'y_end'])

        # save the matching tiles
        unique_removed_tiles.to_csv(Path(patient_data_save_path, "removed_matching_tiles.csv"), index=False)
        unique_original_tiles.to_csv(Path(patient_data_save_path, "original_matching_tiles.csv"), index=False)

        # plot the matching tiles on the original biopsies using matplotlib and seaborn

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.scatterplot(data=pre_tx, x="X_centroid", y="Y_centroid", ax=ax[0])
        for i, row in removed_pre_matching_tiles.iterrows():
            x = row["x_start"]
            y = row["y_start"]
            width = row["x_end"] - row["x_start"]
            height = row["y_end"] - row["y_start"]
            ax[0].add_patch(plt.Rectangle((x, y), width, height, edgecolor='green', facecolor='none'))

        sns.scatterplot(data=post_tx, x="X_centroid", y="Y_centroid", ax=ax[1])
        for i, row in removed_post_matching_tiles.iterrows():
            x = row["x_start"]
            y = row["y_start"]
            width = row["x_end"] - row["x_start"]
            height = row["y_end"] - row["y_start"]
            ax[1].add_patch(plt.Rectangle((x, y), width, height, edgecolor='green', facecolor='none'))

        plt.suptitle(f"Correctly classified tiles over all markers for patient {patient}")
        plt.tight_layout()
        plt.savefig(Path(patient_figure_save_path, f"removed_matching_tiles.png"), dpi=300)
        plt.close('all')

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.scatterplot(data=pre_tx, x="X_centroid", y="Y_centroid", ax=ax[0])
        for i, row in original_pre_matching_tiles.iterrows():
            x = row["x_start"]
            y = row["y_start"]
            width = row["x_end"] - row["x_start"]
            height = row["y_end"] - row["y_start"]
            ax[0].add_patch(plt.Rectangle((x, y), width, height, edgecolor='green', facecolor='none'))
            # set x axis name to X
            ax[0].set_xlabel("X")
            ax[0].set_ylabel("Y")

        sns.scatterplot(data=post_tx, x="X_centroid", y="Y_centroid", ax=ax[1])
        for i, row in original_post_matching_tiles.iterrows():
            x = row["x_start"]
            y = row["y_start"]
            width = row["x_end"] - row["x_start"]
            height = row["y_end"] - row["y_start"]
            ax[1].add_patch(plt.Rectangle((x, y), width, height, edgecolor='green', facecolor='none'))
            ax[1].set_xlabel("X")
            ax[1].set_ylabel("Y")

        plt.suptitle(f"Correctly classified tiles over all markers for patient {patient.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(Path(patient_figure_save_path, f"original_matching_tiles.png"), dpi=300)
        plt.close('all')
