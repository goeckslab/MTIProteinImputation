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
from sklearn.model_selection import cross_val_score
import argparse

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ImplicitModificationWarning)

file_name = "metrics.csv"

# Constants
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
# BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
BIOPSIES = ["9_14_1", "9_14_2", "9_15_1", "9_15_2"]
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


def process_biopsy(biopsy, phenotype):
    try:
        print(f"Processing Biopsy: {biopsy}")
        data = pd.read_csv(f"data/bxs/{biopsy}.csv")
        data = data[SHARED_MARKERS]
        original_data: ad.AnnData = ad.AnnData(data)
        original_data.obs["imageid"] = 1

        imp_data = load_imputed_data(biopsy)

        # Rescale data
        original_data = sm.pp.rescale(original_data, method="standard", verbose=False)

        # Remove imageid from original_data.obs
        original_data.obs.drop("imageid", axis=1, inplace=True)

        # Process original data
        original_data: ad.AnnData = sm.tl.phenotype_cells(original_data, phenotype=phenotype, gate=0.5,
                                                          label="phenotype", verbose=False)

        # fill na of phenotypes with Unknown
        original_data.obs["phenotype"] = original_data.obs["phenotype"].fillna("Unknown")

        for protein in imp_data.columns:
            if protein not in SHARED_MARKERS:
                continue

            tmp_data = data.copy()
            tmp_data[protein] = imp_data[protein]
            imp_ad: ad.AnnData = ad.AnnData(tmp_data)

            # Rescale data
            imp_ad.obs["imageid"] = 1
            imp_ad = sm.pp.rescale(imp_ad, method="standard", verbose=False)
            # Remove imageid
            imp_ad.obs.drop("imageid", axis=1, inplace=True)

            imp_ad = sm.tl.phenotype_cells(imp_ad, phenotype=phenotype, gate=0.5, label="phenotype", verbose=False)

            # Calculate silhouette scores (CPU-bound task)
            # You can calculate the silhouette score before and after imputation for each cell cluster.
            # This method will evaluate how well each cell belongs to its predicted phenotype cluster.
            original_silhouette_score = silhouette_score(original_data.X, original_data.obs["phenotype"])
            imp_silhouette_score = silhouette_score(imp_ad.X, imp_ad.obs["phenotype"])

            # Calculate ARI between original and imputed phenotype calls
            # You can compare how stable the clustering assignments are before and after imputation by calculating the Adjusted Rand Index (ARI)
            # between clusters from the original and imputed datasets.
            # This will show how similar the phenotype assignments are between both datasets.
            ari = adjusted_rand_score(original_data.obs["phenotype"], imp_ad.obs["phenotype"])

            # For original data
            # You can check for compactness (intra-cluster distance) of the phenotype groups.
            # A decrease in intra-cluster variance after imputation might indicate improved phenotype resolution.
            original_compactness = compute_cluster_compactness(original_data.X, original_data.obs["phenotype"])

            # For imputed data
            imputed_compactness = compute_cluster_compactness(imp_ad.X, imp_ad.obs["phenotype"])

            # Use phenotype as the target and data as features
            # You can evaluate how well the features (proteins) discriminate between phenotypes.
            # You could use a classifier (e.g., random forest) to evaluate phenotype separability before and after imputation.
            clf = RandomForestClassifier()

            # Cross-validation score for original data
            original_cv_score = cross_val_score(clf, original_data.X, original_data.obs["phenotype"], cv=5).mean()

            # Cross-validation score for imputed data
            imputed_cv_score = cross_val_score(clf, imp_ad.X, imp_ad.obs["phenotype"], cv=5).mean()

            # print(f"Biopsy: {biopsy}, Protein: {protein}, "
            #      f"Original Silhouette Score: {original_silhouette_score}, "
            #      f"Imputed Silhouette Score: {imp_silhouette_score}, "
            #      f"ARI Score: {ari},"
            #      f"Original Compactness: {original_compactness},"
            #      f"Imputed Compactness: {imputed_compactness},"
            #      f"Original CV Score: {original_cv_score},"
            #      f"Imputed CV Score: {imputed_cv_score},"
            #      f"Imputed LGBM Accuracy: {imputed_accuracy},"
            #      f"Imputed LGBM AUC: {imputed_auc},"
            #      f"Imputed LGBM F1: {imputed_f1},"
            #     f"Original LGBM Accuracy: {original_accuracy},"
            #     f"Original LGBM AUC: {original_auc},"
            #     f"Original LGBM F1: {original_f1},")

            # Prepare the results to append to file
            result = {
                "Biopsy": biopsy,
                "Protein": protein,
                "Original Silhouette Score": original_silhouette_score,
                "Imputed Silhouette Score": imp_silhouette_score,
                "ARI Score": ari,
                "Original Compactness Score": original_compactness,
                "Imputed Compactness Score": imputed_compactness,
                "Original CV Score": original_cv_score,
                "Imputed CV Score": imputed_cv_score,
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
