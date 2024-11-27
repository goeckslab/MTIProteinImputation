from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

save_folder = Path("figures", "supplements", "correlation")

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

    # for each marker in each biopsy calculate correlation between the original and imputed data
    correlations = []
    for biopsy in BIOPSIES:
        original = original_data[biopsy]
        imputed = imputed_data[biopsy]
        for marker in SHARED_MARKERS:
            correlation = original[marker].corr(imputed[marker])
            correlations.append({"Biopsy": biopsy, "Marker": marker, "Correlation": correlation})

    correlations = pd.DataFrame(correlations)
    mean_correlations = pd.DataFrame(correlations.groupby("Marker")["Correlation"].mean()).reset_index()
    print(mean_correlations.mean())

    x = "Marker"
    ax = sns.barplot(data=mean_correlations, x=x, y="Correlation", palette="tab20")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Protein")
    ax.set_title("Correlation between original and imputed protein expression")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(save_folder, "correlation_original_imputed.png"), dpi=300)
    plt.close('all')

    # plot scatter plot for protein biopsy 9_2_1 original vs imputed

    protein_biopsy_map = {
        "AR": "9_3_1",
        "aSMA": "9_14_2",
        "CD45": "9_14_2",
        "CK14": "9_14_2",
        "CK17": "9_14_1",
        "CK19": "9_15_1",
        "Ecad": "9_2_1",
        "EGFR": "9_2_1",
        "ER": "9_2_1",
        "HER2": "9_2_1",
        "Ki67": "9_15_1",
        "p21": "9_15_1",
        "pERK": "9_14_2",
        "PR": "9_15_1",
        "pRB": "9_14_2",
        "Vimentin": "9_15_1",
    }

    for protein, biopsy in protein_biopsy_map.items():
        original = original_data[biopsy]
        imputed = imputed_data[biopsy]

        # scale data
        scaler = MinMaxScaler()
        original = pd.DataFrame(scaler.fit_transform(original), columns=original.columns)
        imputed = pd.DataFrame(scaler.fit_transform(imputed), columns=imputed.columns)

        protein_data = pd.DataFrame({"Original": original[protein], "Imputed": imputed[protein]})
        # Scatterplot with regression line and enhancements
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        sns.scatterplot(data=protein_data, x="Original", y="Imputed", s=3, alpha=0.6, color="blue", ax=ax)

        # Add line of perfect agreement (y = x)
        ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--')
        # Add correlation coefficient
        #r, _ = pearsonr(protein_data["Original"], protein_data["Imputed"])
        #ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

        # Set labels and title
        ax.set_xlabel(f"Original {protein} Expression")
        ax.set_ylabel(f"Imputed {protein} Expression")
        ax.set_title(f"Original vs Imputed {protein} Expression (n={len(protein_data)})")
        # Gridlines and tight layout
        ax.grid(True, linestyle='--', alpha=0.5)
        # limit x and y axis to 0-1
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # add legend with biopsy
        ax.legend([' '.join(biopsy.split('_'))], loc='upper left', title='Biopsy')
        plt.tight_layout()
        plt.savefig(Path(save_folder, f"scatter_original_imputed_{protein}.png"), dpi=300)
        plt.close('all')
