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
import argparse
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import adjusted_mutual_info_score, jaccard_score

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ImplicitModificationWarning)

# Constants
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
PROTEINS_OF_INTEREST = ["aSMA", "CD45", "CK19", "CK14", "CK17"]
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
save_folder = Path("results", "phenotypes")
file_name = "patient_metrics.csv"

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
    test_df["phenotype"] = test_target.values

    # Setup the experiment
    exp = ClassificationExperiment()
    exp.setup(data=train_data, target="phenotype", verbose=False, normalize=True, fold=3)
    clf = exp.create_model("lightgbm", verbose=False)
    predictions = exp.predict_model(clf, data=test_df, verbose=False)

    metrics = exp.pull()
    return metrics["Accuracy"][0]


def process_biopsy(biopsy, phenotype):
    results = []
    try:
        print(f"Processing Biopsy: {biopsy}")
        # Ensure the output file does not exist, so the header is written only once

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
            if protein not in PROTEINS_OF_INTEREST:
                continue

            for i in range(30):
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

                # Prepare the results to append to file
                protein_result = {
                    "Biopsy": biopsy,
                    "Protein": protein,
                    "Original Silhouette Score": original_silhouette_score,
                    "Imputed Silhouette Score": imp_silhouette_score,
                    "ARI Score": ari,
                    "AMI": ami,
                    "Jaccard": jaccard,
                }

                print(protein_result)

                results.append(protein_result)

        return results

    except Exception as e:
        print(f"Error processing biopsy {biopsy}: {e}")
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--iterations", "-i", type=int, default=1)
    args = parser.parse_args()

    workers: int = args.workers
    iterations: int = args.iterations

    print(f"Using {workers} workers for parallel processing.")
    print(f"Running {iterations} iterations for each biopsy.")

    phenotype = pd.read_csv("data/tumor_phenotypes.csv")

    # Remove CK7 from phenotype
    phenotype = phenotype.drop("CK7", axis=1)

    all_results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all biopsies for processing
        # futures = {executor.submit(process_biopsy, biopsy, phenotype): biopsy for biopsy in BIOPSIES}
        futures = {executor.submit(process_biopsy, biopsy, phenotype): biopsy for _ in range(iterations) for biopsy in
                   BIOPSIES}

        try:
            for future in as_completed(futures):
                biopsy = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.extend(result)
                        print(f"Biopsy {biopsy} processed successfully.")
                    else:
                        print(f"Biopsy {biopsy} returned no results.")
                except Exception as e:
                    print(f"Biopsy {biopsy} generated an exception: {e}")

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping execution and saving scores.")
            # Optionally, handle partial results here
            # For simplicity, just exit
            sys.exit(0)

    # After all biopsies are processed, save the aggregated results
    if all_results:
        save_file = Path(save_folder, file_name)
        try:
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(save_file, index=False)
            print(f"All results saved to '{save_file}'.")
        except Exception as e:
            print(f"Error saving results to '{save_file}': {e}")
    else:
        print("No results to save.")
