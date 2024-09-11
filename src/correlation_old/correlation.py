from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    pre:bool = True if "1" == biopsy.split('_')[2] else False

    if biopsy == "9_14_2" or biopsy == "9_15_2":
        assert pre == False, "Patient 9_14_2 and 9_15_2 are a post biopsies"

    patient = '_'.join(biopsy.split('_')[:2])


    original_data = pd.read_csv(f"data/bxs/{biopsy}.csv", nrows=1000)
    if pre:
        imputed_data = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/pre_treatment.csv", nrows=1000)
    else:
        imputed_data = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/on_treatment.csv", nrows=1000)

    original_data = original_data[SHARED_MARKERS]
    imputed_data = imputed_data[SHARED_MARKERS]


    # normalize data using sklearn
    original_data = pd.DataFrame(StandardScaler().fit_transform(original_data), columns=original_data.columns)
    imputed_data = pd.DataFrame(StandardScaler().fit_transform(imputed_data), columns=imputed_data.columns)

    # Calculate Pearson correlation for each cell (row-wise correlation)
    cell_wise_correlations = original_data.T.corrwith(imputed_data.T, axis=0)

    print(cell_wise_correlations)

    # Assuming 'cell_wise_correlations' is your Series of correlation values for each cell
    plt.figure(figsize=(8, 6))
    plt.hist(cell_wise_correlations, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Cell-Wise Correlations')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Number of Cells')
    plt.show()
    plt.close('all')

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=cell_wise_correlations, orient='h', inner="quartile", color='skyblue')
    plt.title('Violin Plot of Cell-Wise Correlations')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.show()
    plt.close('all')

    protein_correlations = original_data.corrwith(imputed_data, axis=0)

    correlation_matrix_true = original_data.corr()
    correlation_matrix_imputed = imputed_data.corr()

    # Compute differences between correlation matrices
    difference_matrix = correlation_matrix_imputed - correlation_matrix_true

    plt.figure(figsize=(10, 8))
    sns.heatmap(difference_matrix, cmap='coolwarm', annot=True)
    plt.title('Difference in Protein-Protein Correlations (Imputed - True)')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(original_data.values.flatten(), imputed_data.values.flatten(), alpha=0.5)
    plt.title('Scatter Plot of Ground Truth vs. Imputed Values')
    plt.xlabel('Ground Truth Expression')
    plt.ylabel('Imputed Expression')
    #plt.show()
