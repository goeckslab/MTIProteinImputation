from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

save_folder = Path("results", "evaluation")

if __name__ == '__main__':

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    original_data = {}
    imputed_data = {}
    for biopsy in BIOPSIES:
        patient = '_'.join(biopsy.split('_')[:2])
        pre: bool = True if "1" == biopsy.split('_')[2] else False

        if biopsy == "9_14_2" or biopsy == "9_15_2":
            assert pre == False, "Patient 9_14_2 and 9_15_2 are post biopsies"

        original = pd.read_csv(f"data/bxs/{biopsy}.csv")
        if pre:
            imputed = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/pre_treatment.csv")
        else:
            imputed = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/on_treatment.csv")

        original = original[SHARED_MARKERS]
        imputed = imputed[SHARED_MARKERS]

        original_data[biopsy] = original
        imputed_data[biopsy] = imputed

    results = []

    # calculate cluster for original data for each protein
    for biopsy in BIOPSIES:
        print(f"Biopsy: {biopsy}")
        original = original_data[biopsy]
        imputed_markers = imputed_data[biopsy]
        for marker in SHARED_MARKERS:
            print(f"Marker: {marker}")
            # scale the data
            original_scaled = pd.DataFrame(MinMaxScaler().fit_transform(original), columns=original.columns)
            imputed_scaled = pd.DataFrame(MinMaxScaler().fit_transform(imputed_markers),
                                          columns=imputed_markers.columns)

            # Initialize and fit KMeans
            kmeans = KMeans(random_state=42, n_clusters=6)
            kmeans.fit(original_scaled)

            # Get cluster labels
            cluster_labels = kmeans.labels_

            og_replaced_data = original_scaled.copy()
            og_replaced_data[marker] = imputed_scaled[marker]

            # assert that imputed data is not the same as original data
            assert not og_replaced_data[marker].equals(
                original_scaled[marker]), "Imputed data is the same as original data"

            # Cluster the modified (imputed) data
            kmeans.fit(og_replaced_data)
            imp_cluster_labels = kmeans.labels_

            # Calculate Adjusted Rand Index
            ari = adjusted_rand_score(cluster_labels, imp_cluster_labels)

            # Calculate Silhouette Scores
            silhouette_original = silhouette_score(original_scaled, cluster_labels)
            silhouette_imputed = silhouette_score(og_replaced_data, imp_cluster_labels)

            results.append({"Biopsy": biopsy, "Marker": marker, "ARI": ari,
                            "Silhouette Original": silhouette_original, "Silhouette Imputed": silhouette_imputed})

    results_df = pd.DataFrame(results)
    # save
    results_df.to_csv(Path(save_folder, "cluster_metrics.csv"), index=False)
