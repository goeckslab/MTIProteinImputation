from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
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

    original_data = pd.read_csv(f"data/bxs/{biopsy}.csv", nrows=10000)
    if pre:
        imputed_data = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/pre_treatment.csv", nrows=10000)
    else:
        imputed_data = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/on_treatment.csv", nrows=10000)

    original_data = original_data[SHARED_MARKERS]
    imputed_data = imputed_data[SHARED_MARKERS]

    # Normalize data using sklearn
    original_data = pd.DataFrame(StandardScaler().fit_transform(original_data), columns=original_data.columns)
    imputed_data = pd.DataFrame(StandardScaler().fit_transform(imputed_data), columns=imputed_data.columns)

    # Calculate cosine similarity between proteins
    cosine_sim_original = cosine_similarity(original_data.T)
    cosine_sim_imputed = cosine_similarity(imputed_data.T)

    # Convert the cosine similarity matrices to DataFrames for better readability
    cosine_sim_original_df = pd.DataFrame(cosine_sim_original, index=original_data.columns, columns=original_data.columns)
    cosine_sim_imputed_df = pd.DataFrame(cosine_sim_imputed, index=imputed_data.columns, columns=imputed_data.columns)

    difference = cosine_sim_original_df - cosine_sim_imputed_df

    # Display the cosine similarity matrices
    print("Cosine Similarity (Original Data):")
    print(cosine_sim_original_df)

    print("\nCosine Similarity (Imputed Data):")
    print(cosine_sim_imputed_df)

    # Optionally, visualize the cosine similarity using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_original_df, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity between Proteins (Original Data)')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_imputed_df, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity between Proteins (Imputed Data)')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(difference, annot=True, cmap='coolwarm')
    plt.title('Difference in Cosine Similarity between Proteins')
    plt.show()