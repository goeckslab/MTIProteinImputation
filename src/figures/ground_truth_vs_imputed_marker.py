import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging
from scipy.stats import ks_2samp
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def setup_log_file(save_path: Path):
    save_file = Path(save_path, "debug.log")

    if save_file.exists():
        save_file.unlink()

    file_logger = logging.FileHandler(save_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for handler in log.handlers[:]:  # remove all old handlers
        log.removeHandler(handler)
    log.addHandler(file_logger)
    log.addHandler(logging.StreamHandler())


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


def load_train_data():
    base_path = Path("data", "bxs") if spatial == 0 else Path("data", f"bxs_{spatial}_µm")
    train_data = []
    for file in os.listdir(base_path):
        file_name = Path(file).stem
        if file.endswith(".csv") and patient not in file_name:
            print("Loading train file: " + file)
            data = pd.read_csv(Path(base_path, file))
            data = clean_column_names(data)
            train_data.append(data)

    return pd.concat(train_data)


if __name__ == '__main__':
    save_path: Path = Path("figures", "supplements", "ground_truth_vs_imputed")

    argparse = ArgumentParser()
    argparse.add_argument("--biopsy", "-b", help="the biopsy used. Should be just 9_2_1", required=True)
    argparse.add_argument("-sp", "--spatial", action="store", help="The spatial radius used",
                          choices=[0, 23, 46, 92, 138, 184], type=int, default=0)
    # argparse.add_argument("--mode", choices=["ip", "exp"], default="ip", help="the mode used")
    argparse.add_argument("--model", choices=["EN", "LGBM", "AE", "AE M"], help="the model used",
                          required=True)
    args = argparse.parse_args()

    biopsy: str = args.biopsy
    # mode: str = args.mode
    model: str = args.model
    spatial: int = args.spatial
    patient: str = '_'.join(biopsy.split("_")[:2])

    save_path = Path(save_path, model, biopsy, str(spatial))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    setup_log_file(save_path=save_path)

    logging.debug(f"Biopsy: {biopsy}")
    logging.debug(f"Model: {model}")
    logging.debug(f"Patient: {patient}")
    logging.debug(f"Spatial: {spatial}")

    assert patient in biopsy, "The biopsy should be of the form 9_2_1, where 9_2 is the patient and 1 is the biopsy. Patient should be in biopsy"

    if model == "AE":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "bxs", f"{biopsy}.csv"))
        ground_truth = ground_truth[SHARED_MARKERS]
        # scale ground truth using min max sklearn min max scaler
        ground_truth = pd.DataFrame(MinMaxScaler().fit_transform(ground_truth), columns=ground_truth.columns)

        predictions_load_path: Path = Path("src", "ae", "single_imputation", "exp", "mean", biopsy, "0",
                                           "experiment_run_0")

        predictions = []
        for i in range(5, 10):
            prediction = pd.read_csv(Path(predictions_load_path, f"{i}_predictions.csv"))
            predictions.append(prediction)

        assert len(predictions) == 5, "Number of predictions is not 5"
        predictions = pd.concat(predictions).groupby(level=0).mean()

        train_data: pd.DataFrame = load_train_data()
        train_data = train_data[SHARED_MARKERS]
        # scale
        train_data = pd.DataFrame(MinMaxScaler().fit_transform(train_data), columns=train_data.columns)

    elif model == "AE M":
        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")
        predictions: pd.DataFrame = pd.read_csv(Path("data", "cleaned_data", "predictions", "ae_m", "predictions.csv"),
                                                sep=",")
        train_data: pd.DataFrame = pd.read_csv(
            Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
            sep="\t")



    else:
        raise ValueError("Model not recognized")

    for protein in predictions.columns:
        pred = predictions[protein]
        gt = ground_truth[protein]
        train = train_data[protein]

        sns.histplot(pred, color="orange", label="Imputed", kde=True, element="poly")
        # log y-axis
        plt.yscale('log')

        sns.histplot(gt, color="blue", label="Ground Truth", kde=True, element="poly")
        sns.histplot(train, color="green", label="Train", kde=True, element="poly")

        # change y axis label to cell count
        plt.ylabel("Cell Count")
        plt.xlabel(f"{protein} Expression")
        plt.legend()
        plt.savefig(Path(save_path, f"{protein}.png"), dpi=300, bbox_inches='tight')
        plt.close('all')
