from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SAVE_FOLDER = Path("results", "evaluation")
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--protein", "-p", required=True, type=str)
    parser.add_argument("--biopsy", "-b", required=True, type=str)

    args = parser.parse_args()

    protein:str = args.protein
    biopsy: str = args.biopsy
    patient: str = "_".join(biopsy.split("_")[:2])
    isPre:bool = True if biopsy.split("_")[2] == "1" else False

    save_path = Path(SAVE_FOLDER, biopsy)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    biopsy_df = pd.read_csv(Path("data", "bxs", f"{biopsy}.csv"))
    imputed_df = pd.read_csv(Path("results", "imputed_data", "ae", "single", "exp", patient, "0",
                                  "pre_treatment.csv" if isPre else "on_treatment.csv"))

    og_data = pd.DataFrame({
        'X': biopsy_df['X_centroid'],
        'Y': biopsy_df['Y_centroid'],
        'Original Expression': biopsy_df[protein],

    })

    imp_data = pd.DataFrame({
        'X': biopsy_df['X_centroid'],
        'Y': biopsy_df['Y_centroid'],
        'Imputed Expression': imputed_df[protein]
    })

    # scale the data
    og_data['Original Expression'] = MinMaxScaler().fit_transform(og_data['Original Expression'].values.reshape(-1, 1))
    imp_data['Imputed Expression'] = MinMaxScaler().fit_transform(imp_data['Imputed Expression'].values.reshape(-1, 1))

    # save the data
    og_data.to_csv(Path(save_path, f"{biopsy}_{protein}_original.csv"), index=False)
    imp_data.to_csv(Path(save_path, f"{biopsy}_{protein}_imputed.csv"), index=False)

