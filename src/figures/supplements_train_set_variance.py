import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import logging

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("train_set_variance.log"),
                        logging.StreamHandler()
                    ])


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


if __name__ == '__main__':
    base_path: Path = Path("images", "supplements", "training_distribution")

    if not base_path.exists():
        base_path.mkdir(parents=True)

    setup_log_file(save_path=base_path)

    for patient in PATIENTS:
        save_path = Path(base_path, patient)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        ground_truth: pd.DataFrame = pd.read_csv(
            Path("data", "bxs", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"), sep="\t")

        ground_truth = ground_truth.loc[(ground_truth != 0.0).any(axis=1)]

        variance_scores = []
        for protein in ground_truth.columns:
            gt = ground_truth[protein]
            sns.histplot(gt, color="blue", label="Expression", kde=True)
            plt.ylabel("Cell Count")
            plt.xlabel(f"{protein} Expression")
            plt.savefig(Path(save_path, f"{protein}.png"), dpi=300)
            # close figure
            plt.close()

        # plot violin plot for each biopsy
        fig = plt.figure(figsize=(10, 5), dpi=300)
        sns.violinplot(data=ground_truth)
        plt.title(patient.replace("_", " "))
        plt.savefig(Path(save_path, f"{patient}.png"), dpi=300)
        plt.close()
