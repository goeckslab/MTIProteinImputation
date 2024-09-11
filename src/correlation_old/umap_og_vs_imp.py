import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path
from argparse import ArgumentParser
import seaborn as sns

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument("--biopsy", "-b", type=str, help="The patient to use", required=True,
                        choices=["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"])

    args = parser.parse_args()

    biopsy = args.biopsy
    pre: bool = True if "1" == biopsy.split('_')[2] else False

    if biopsy == "9_14_2" or biopsy == "9_15_2":
        assert pre == False, "Patient 9_14_2 and 9_15_2 are post biopsies"

    patient = '_'.join(biopsy.split('_')[:2])

    original_data = pd.read_csv(f"data/bxs/{biopsy}.csv", nrows=1000)
    if pre:
        imputed_data = pd.read_csv(Path(f"results/imputed_data/ae/single/exp/{patient}/0/pre_treatment.csv"), nrows=1000)
    else:
        imputed_data = pd.read_csv(Path(f"results/imputed_data/ae/single/exp/{patient}/0/on_treatment.csv"), nrows=1000)

    original_data = original_data[SHARED_MARKERS]
    imputed_data = imputed_data[SHARED_MARKERS]

    # Normalize data using sklearn
    scaler = StandardScaler()
    original_data_scaled = scaler.fit_transform(original_data)
    imputed_data_scaled = scaler.fit_transform(imputed_data)

    original_data_scaled = pd.DataFrame(original_data_scaled, columns=original_data.columns)
    imputed_data_scaled = pd.DataFrame(imputed_data_scaled, columns=imputed_data.columns)

    original_data_scaled["Type"] = "Original"
    imputed_data_scaled["Type"] = "Imputed"

    print(original_data_scaled.shape)
    print(imputed_data_scaled.shape)

    combined = pd.concat([original_data_scaled, imputed_data_scaled], ignore_index=False)


    types = combined["Type"].copy()

    expression_data = combined.drop(columns=["Type"])
    # Apply UMAP to reduce dimensions to 2D for protein clustering
    umap_reducer = umap.UMAP(n_components=2)

    # Concatenate the data for combined UMAP embedding
    umap_embedding = umap_reducer.fit_transform(expression_data)

    # Add UMAP coordinates to combined data
    combined['UMAP1'] = umap_embedding[:, 0]
    combined['UMAP2'] = umap_embedding[:, 1]

    # Plot the UMAP embeddings
    plt.figure(figsize=(12, 6))

    # Plot for original data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='UMAP1', y='UMAP2', data=combined[combined['Type'] == 'Original'], palette='tab20',
                    legend=False)
    plt.title('UMAP Projection of Protein Clustering (Original Data)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Plot for imputed data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='UMAP1', y='UMAP2', data=combined[combined['Type'] == 'Imputed'],
                    palette='tab20', legend='full')
    plt.title('UMAP Projection of Protein Clustering (Imputed Data)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.tight_layout()
    plt.show()
