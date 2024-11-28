import scimap as sm
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import ImplicitModificationWarning
from sklearn.metrics import silhouette_score
# surpress future wanrings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ImplicitModificationWarning)

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
save_folder = Path("results", "phenotypes")


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


if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    scores = []
    try:
        for biopsy in BIOPSIES:
            print(f"Biopsy: {biopsy}")
            data = pd.read_csv(f"data/bxs/{biopsy}.csv")
            data = data[SHARED_MARKERS]
            original_data: ad.AnnData = ad.AnnData(data)
            original_data.obs["imageid"] = 1
            phenotype = pd.read_csv("data/tumor_phenotypes.csv")

            # remove CK7 from phenotype
            phenotype = phenotype.drop("CK7", axis=1)

            imp_data = load_imputed_data(biopsy)

            # rescale dat
            # original_data = sm.pp.rescale(original_data, method="standard")
            # drop the imageid column#
            original_data: ad.AnnData = sm.tl.phenotype_cells(original_data, phenotype=phenotype, gate=0.5,
                                                              label="phenotype", verbose=False)
            print(original_data.obs["phenotype"].unique())

            for protein in imp_data.columns:
                print(f"Protein: {protein}")
                tmp_data = data.copy()
                tmp_data[protein] = imp_data[protein]
                imp_ad = ad.AnnData(tmp_data)

                # imp_ad = sm.pp.rescale(imp_ad, method="standard")
                imp_ad = sm.tl.phenotype_cells(imp_ad, phenotype=phenotype, gate=0.5, label="phenotype", verbose=False)

                # For original data
                #original_silhouette_score = silhouette_score(original_data.X, original_data.obs["phenotype"])

                # For imputed data
                #imp_silhouette_score = silhouette_score(imp_ad.X, imp_ad.obs["phenotype"])

                original_variance = original_data.X.var(axis=0)
                imputed_variance = imp_ad.X.var(axis=0)

                print("Variance reduction:", original_variance - imputed_variance)
                input()

                print(f"Biopsy: {biopsy}, Protein: {protein}")
                #print(f"Original Silhouette Score: {original_silhouette_score}")
                #print(f"Imputed Silhouette Score: {imp_silhouette_score}")
                scores.append(
                    {"Biopsy": biopsy, "Protein": protein, "Original Silhouette Score": 0,
                     "Imputed Silhouette Score": 0})

        scores = pd.DataFrame(scores)
        # scores.to_csv(Path(save_folder, "silhouette.csv"), index=False)


    except KeyboardInterrupt:
        scores = pd.DataFrame(scores)
        # scores.to_csv(Path(save_folder, "silhouette.csv"), index=False)
