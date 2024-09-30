from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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
    print(mean_correlations)

    x = "Marker"
    ax = sns.barplot(data=mean_correlations, x=x, y="Correlation", palette="tab20")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Protein")
    ax.set_title("Correlation between original and imputed protein expression")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(save_folder, "correlation_original_imputed.png"), dpi=300)
