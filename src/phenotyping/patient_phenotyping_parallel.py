import scimap as sm
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import ImplicitModificationWarning
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import signal
import sys
import threading
from sklearn.metrics import adjusted_rand_score
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier
import argparse
from sklearn.metrics import accuracy_score
from pycaret.classification import ClassificationExperiment
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_mutual_info_score, jaccard_score
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ImplicitModificationWarning)

file_name = "patient_metrics.csv"

# Constants
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
save_folder = Path("results", "phenotypes")

# List to collect scores and thread lock for thread safety
lock = threading.Lock()

# Create folder if it doesn't exist
if not save_folder.exists():
    save_folder.mkdir(parents=True)


def compute_cluster_compactness(data, labels):
    compactness = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_data = data[labels == label]
        if cluster_data.shape[0] > 1:
            compactness.append(np.mean(pdist(cluster_data)))

    return np.mean(compactness)


def load_imputed_data(biopsy: str):
    patient = '_'.join(biopsy.split("_")[0:2])
    pre: bool = True if "1" == biopsy.split("_")[-1] else False
    if pre:
        imp_treatment = pd.read_csv(
            Path("results", "imputed_data", "ae", "single", "exp", patient, "0", "pre_treatment.csv"))
    else:
        imp_treatment = pd.read_csv(
            Path("results", "imputed_data", "ae", "single", "exp", patient, "0", "on_treatment.csv"))

    return imp_treatment


def load_train_data(biopsy: str, phenotype: pd.DataFrame):
    try:
        bxs = []
        for file in Path("data", "bxs").glob("*.csv"):
            patient = '_'.join(biopsy.split("_")[0:2])
            if patient in file.stem:
                continue

            df = pd.read_csv(file)
            df = df[SHARED_MARKERS]
            df = ad.AnnData(df)
            df.obs["imageid"] = file.stem
            df = sm.pp.rescale(df, method="standard", verbose=False)
            df = sm.tl.phenotype_cells(df, phenotype=phenotype, gate=0.5, label="phenotype", verbose=False)

            new_df = pd.DataFrame(df.X, columns=SHARED_MARKERS)
            new_df["phenotype"] = df.obs["phenotype"].values
            bxs.append(new_df)

        df = pd.concat(bxs)

        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading train data for biopsy {biopsy}: {e}")
        sys.exit(0)


def run_lgbm(train_data, test_features, test_target):
    test_df = pd.DataFrame(test_features, columns=SHARED_MARKERS)
    test_df["phenotype"] = test_target

    # Setup the experiment
    exp = ClassificationExperiment()
    exp.setup(data=train_data, target="phenotype", verbose=False, normalize=True, fold=3)
    clf = exp.create_model("lightgbm", verbose=False)
    predictions = exp.predict_model(clf, data=test_features, verbose=False)

    metrics = exp.pull()
    return metrics["Accuracy"][0]


def process_biopsy(biopsy, phenotype):
    try:
        print(f"Processing Biopsy: {biopsy}")

        train_data = load_train_data(biopsy, phenotype)

        test_data = pd.read_csv(f"data/bxs/{biopsy}.csv")
        test_data = test_data[SHARED_MARKERS]
        original_test_data: ad.AnnData = ad.AnnData(test_data)
        original_test_data.obs["imageid"] = 1
        imp_data = load_imputed_data(biopsy)

        # Rescale data
        original_test_data = sm.pp.rescale(original_test_data, method="standard", verbose=False)

        # Process original data
        original_test_data: ad.AnnData = sm.tl.phenotype_cells(original_test_data, phenotype=phenotype, gate=0.5,
                                                               label="phenotype", verbose=False)

        # fill na of phenotypes with Unknown

        original_test_data.obs["phenotype"] = original_test_data.obs["phenotype"].fillna("Unknown")

        # Calculate silhouette score for original data
        original_silhouette_score = silhouette_score(original_test_data.X, original_test_data.obs["phenotype"])

        for protein in imp_data.columns:
            if protein not in SHARED_MARKERS:
                continue

            tmp_data = test_data.copy()
            tmp_data[protein] = imp_data[protein]
            imp_ad: ad.AnnData = ad.AnnData(tmp_data)
            # Rescale data
            imp_ad.obs["imageid"] = 1
            imp_ad = sm.pp.rescale(imp_ad, method="standard", verbose=False)

            imp_ad = sm.tl.phenotype_cells(imp_ad, phenotype=phenotype, gate=0.5, label="phenotype", verbose=False)

            # Calculate silhouette scores (CPU-bound task)
            # You can calculate the silhouette score before and after imputation for each cell cluster.
            # This method will evaluate how well each cell belongs to its predicted phenotype cluster.

            imp_silhouette_score = silhouette_score(imp_ad.X, imp_ad.obs["phenotype"])

            # Calculate ARI between original and imputed phenotype calls
            # You can compare how stable the clustering assignments are before and after imputation by calculating the Adjusted Rand Index (ARI)
            # between clusters from the original and imputed datasets.
            # This will show how similar the phenotype assignments are between both datasets.
            ari = adjusted_rand_score(original_test_data.obs["phenotype"], imp_ad.obs["phenotype"])

            # Calculate AMI between original and imputed phenotype calls
            ami = adjusted_mutual_info_score(original_test_data.obs["phenotype"], imp_ad.obs["phenotype"])
            # Calculate Jaccard between original and imputed phenotype calls
            jaccard = jaccard_score(original_test_data.obs["phenotype"], imp_ad.obs["phenotype"], average='macro')

            # Calculate accuracy or other performance metrics
            original_accuracy = run_lgbm(train_data, pd.DataFrame(original_test_data.X, columns=SHARED_MARKERS),
                                         original_test_data.obs["phenotype"])
            imputed_accuracy = run_lgbm(train_data, pd.DataFrame(imp_ad.X, columns=SHARED_MARKERS),
                                        original_test_data.obs["phenotype"])

            # Prepare the results to append to file
            result = {
                "Biopsy": biopsy,
                "Protein": protein,
                "Original Silhouette Score": original_silhouette_score,
                "Imputed Silhouette Score": imp_silhouette_score,
                "ARI Score": ari,
                "AMI": ami,
                "Jaccard": jaccard,
                "Original CV Score": original_accuracy,
                "Imputed CV Score": imputed_accuracy,
            }

            print(result)

            # Thread-safe file appending
            with lock:
                append_to_file(result)

    except Exception as e:
        print(f"Error processing biopsy {biopsy}: {e}")
        return []


def append_to_file(result):
    """
    Appends a single result as a new row to the CSV file.
    """
    df = pd.DataFrame([result])
    df.to_csv(Path(save_folder, file_name), mode='a', header=not Path(save_folder, file_name).exists(),
              index=False)


def signal_handler(sig, frame):
    print("KeyboardInterrupt received. Saving scores...")
    sys.exit(0)


# Attach the signal handler to gracefully save results on Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", "-w", type=int, default=1)
    args = parser.parse_args()

    workers: int = args.workers

    print(f"Using {workers} workers for parallel processing.")

    # Ensure the output file does not exist, so the header is written only once
    if Path(save_folder, file_name).exists():
        Path(save_folder, file_name).unlink()

    phenotype = pd.read_csv("data/tumor_phenotypes.csv")

    # Remove CK7 from phenotype
    phenotype = phenotype.drop("CK7", axis=1)

    with ProcessPoolExecutor(max_workers=workers) as executor:  # Using ProcessPoolExecutor for CPU-bound tasks
        futures = [executor.submit(process_biopsy, biopsy, phenotype) for biopsy in BIOPSIES]

        try:
            for future in as_completed(futures):
                future.result()  # Wait for each future to complete

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping execution and saving scores.")
            sys.exit(0)
