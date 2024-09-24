import scimap as sm
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import ImplicitModificationWarning
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ImplicitModificationWarning)

# Constants
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
save_folder = Path("results", "phenotypes")

# Create folder if it doesn't exist
if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':

    phenotype = pd.read_csv("data/tumor_phenotypes.csv")

    # Remove CK7 from phenotype
    phenotype = phenotype.drop("CK7", axis=1)
    # Remove CD45 from phenotype
    #phenotype = phenotype.drop("CD45", axis=1)

    for biopsy in BIOPSIES:
        print("Processing biopsy: ", biopsy)
        test_data = pd.read_csv(f"data/bxs/{biopsy}.csv")
        test_data = test_data[SHARED_MARKERS]
        original_test_data: ad.AnnData = ad.AnnData(test_data)
        original_test_data.obs["imageid"] = 1

        # Rescale data
        original_test_data = sm.pp.rescale(original_test_data, method="standard", verbose=False)

        # Process original data
        original_test_data: ad.AnnData = sm.tl.phenotype_cells(original_test_data, phenotype=phenotype, gate=0.5,
                                                               label="phenotype", verbose=False)

        # save the phenotype data
        original_test_data.obs.to_csv(Path(save_folder, f"{biopsy}_phenotype.csv"))
