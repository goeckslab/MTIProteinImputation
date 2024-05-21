import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_PATH = Path("figures", "tiles")

SHARED_MARKERS = ["pRB", "CD45", "CK19", "Ki67", "aSMA", "Ecad", "PR", "CK14", "HER2", "AR", "CK17", "p21", "Vimentin",
                  "pERK", "EGFR", "ER"]
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

if __name__ == '__main__':

    for patient in PATIENTS:
        # Load the data
        pre_tx_path = Path("data", "bxs", f"{patient}_1.csv")
        post_tx_path = Path("data", "bxs", f"{patient}_2.csv")

        pre_tx = pd.read_csv(pre_tx_path)
        post_tx = pd.read_csv(post_tx_path)

        patient_save_path = Path(SAVE_PATH, patient)
        if not patient_save_path.exists():
            patient_save_path.mkdir(parents=True)

        for protein in SHARED_MARKERS:
            imputed_predictions = pd.read_csv(
                Path("results", "classifier", "pycaret_tiles_spatial_information", "exp", patient, "0",
                     f"{protein}_imputed_predictions.csv"))

            # get pre treatment predictions
            pre_tx_predictions = imputed_predictions[imputed_predictions["Treatment"] == "PRE"]
            pre_tx_predictions = pre_tx_predictions[
                ["Treatment", "X_start", "X_end", "Y_start", "Y_end", "prediction_label"]]

            # get post treatment predictions
            post_tx_predictions = imputed_predictions[imputed_predictions["Treatment"] == "ON"]
            post_tx_predictions = post_tx_predictions[
                ["Treatment", "X_start", "X_end", "Y_start", "Y_end", "prediction_label"]]

            # filter only wrong predictions
            pre_tx_predictions = pre_tx_predictions[
                pre_tx_predictions["prediction_label"] != pre_tx_predictions["Treatment"]]
            post_tx_predictions = post_tx_predictions[
                post_tx_predictions["prediction_label"] != post_tx_predictions["Treatment"]]
            # plot the pre-treatment biopsy

            sns.set(style="whitegrid")
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            sns.scatterplot(data=pre_tx, x="X_centroid", y="Y_centroid", ax=ax[0])
            for i, row in pre_tx_predictions.iterrows():
                x = row["X_start"]
                y = row["Y_start"]
                width = row["X_end"] - row["X_start"]
                height = row["Y_end"] - row["Y_start"]
                ax[0].add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor='red', lw=2))
                # add the prediction label
                # ax[0].text(x, y, "O", fontsize=12, color="red", rotation=90)
                # center the text in the rectangle
                # ax[0].text(x + width / 2, y + height / 2, "O", fontsize=12, color="red", rotation=90)

            ax[0].set_title(f"Pre-treatment {protein} biopsy")
            ax[0].set_xlabel("X")
            ax[0].set_ylabel("Y")
            # ax[0].legend(title="Prediction")

            # plot the post-treatment biopsy
            sns.scatterplot(data=post_tx, x="X_centroid", y="Y_centroid", ax=ax[1])
            for i, row in post_tx_predictions.iterrows():
                x = row["X_start"]
                y = row["Y_start"]
                width = row["X_end"] - row["X_start"]
                height = row["Y_end"] - row["Y_start"]
                ax[1].add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor='red', lw=2))
                # add the prediction label
                # ax[1].text(x, y, "P", fontsize=12, color="red", rotation=90)
                # center the text in the rectangle
                # ax[1].text(x + width / 2, y + height / 2, "P", fontsize=12, color="red", rotation=90)

            ax[1].set_title(f"Post-treatment {protein} biopsy")
            ax[1].set_xlabel("X")
            ax[1].set_ylabel("Y")
            # ax[1].legend(title="Prediction")
            # title
            plt.suptitle(f"Biopsy images for {protein} in patient {patient}")
            plt.tight_layout()
            plt.savefig(Path(patient_save_path, f"{protein}_biopsy.png"), dpi=300)
            plt.close('all')
