import argparse, logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

save_path = Path("plots", "figures", "supplements", "predicted_vs_actual")
ground_truth_path = Path("data/bxs/preprocessed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", help="the predicted marker", required=False)
    parser.add_argument("--biopsy", "-b", help="the biopsy used", required=True)
    parser.add_argument("--mode", choices=["ip", "exp"], default="exp", help="the mode used")
    parser.add_argument("--model", choices=["EN", "LGBM", "AE"], help="the model to used", required=True)
    parser.add_argument("--radius", choices=[0, 15, 30, 60, 90, 120], default=0, type=int)
    args = parser.parse_args()

    marker: str = args.marker
    model: str = args.model
    biopsy: str = args.biopsy
    mode: str = str(args.mode).upper()
    radius: int = args.radius
    patient: str = "_".join(biopsy.split("_")[:-1])

    save_path = Path(save_path, model, biopsy, str(radius))

    ground_truth_path: Path = Path(ground_truth_path, f"{biopsy}_preprocessed_dataset.tsv")
    if model == "AE":
        file_name = "on_treatment" if biopsy.split("_")[-1] == "2" else "pre_treatment"
        predicted_path: Path = Path("results", "imputed_data", "ae", "single", mode, patient, str(radius),
                                    f"{file_name}.csv")
    else:
        raise ValueError("Invalid model")

    ground_truth: pd.DataFrame = pd.read_csv(ground_truth_path, delimiter="\t", header=0)
    predictions: pd.DataFrame = pd.read_csv(predicted_path)

    # difference = ground_truth[marker] - predictions[marker]

    # fig = plt.figure(figsize=(5, 3), dpi=200)
    # plt.plot(difference, alpha=0.5, label=marker, marker='o', linestyle='None')
    # plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
    #         np.poly1d(np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(), 1))(
    #             np.unique(ground_truth[[marker]].values.flatten())), color='red')

    # plt.show()
    # plt.close('all')

    print(predictions)
    print(ground_truth)

    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(ground_truth[marker], predictions[marker], alpha=0.5, label=marker)
    plt.plot(np.unique(ground_truth[[marker]].values.flatten()),
             np.poly1d(
                 np.polyfit(ground_truth[[marker]].values.flatten(), predictions[[marker]].values.flatten(),
                            1))(
                 np.unique(ground_truth[[marker]].values.flatten())), color='red')
    # add correlation coefficient
    corr = np.corrcoef(ground_truth[marker], predictions[marker])[0, 1]
    plt.text(0.1, 0.9, f"Correlation: {corr:.2f}", transform=plt.gca().transAxes)
    plt.xlabel(f"Ground Truth {marker}")
    plt.ylabel(f"Imputed {marker}")
    plt.title(f"{marker}: Imputed vs Actual")
    plt.tight_layout()

    if not save_path.exists():
        save_path.mkdir(parents=True)

    #plt.savefig(Path(save_path, f"{marker}_predicted_vs_actual.png"), dpi=300)
    plt.show()