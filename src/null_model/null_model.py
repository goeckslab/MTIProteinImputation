import pandas as pd
from pathlib import Path
import argparse
import os
from tqdm import tqdm

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
save_path = Path("results", "scores", "null_model")


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


def load_train_data(base_path: Path, patient: str):
    train_data = []
    for file in os.listdir(base_path):
        file_name = Path(file).stem
        if file.endswith(".csv") and patient not in file_name:
            print("Loading train file: " + file)
            data = pd.read_csv(Path(base_path, file))
            data = clean_column_names(data)
            train_data.append(data)

    assert len(train_data) == 6, f"There should be 6 train datasets, loaded {len(train_data)}"
    return pd.concat(train_data)


if __name__ == '__main__':

    if not save_path.exists():
        save_path.mkdir(parents=True)

    parser = argparse.ArgumentParser(description='Run null model')
    args = parser.parse_args()
    scores = []
    for biopsy in BIOPSIES:
        print("Working on biopsy: ", biopsy)
        patient = "_".join(biopsy.split("_")[:-1])
        # load biopsy data
        test_data = pd.read_csv(Path("data", "bxs", f"{biopsy}.csv"))
        train_data = load_train_data(Path("data", "bxs"), patient=patient)
        for protein in SHARED_MARKERS:
            for i in tqdm(range(30)):
                # use the mean to impute the values
                #y_hat = train_data.sample(n=len(test_data), replace=True)[protein]
                y_hat = train_data[protein].sample(n=len(test_data), replace=True).mean()

                # calculate mae, rmse
                mae = (y_hat - test_data[protein]).abs().mean()
                rmse = ((y_hat - test_data[protein]) ** 2).mean() ** .5

                # print(f"Biopsy {biopsy}, Protein {protein}, MAE: {mae}, RMSE: {rmse}")
                scores.append({"Biopsy": biopsy, "Protein": protein, "MAE": mae, "RMSE": rmse, "Iteration": i})

    scores = pd.DataFrame(scores)

    # sort scores by protein
    scores = scores.sort_values(by=["Protein", "Biopsy"])
    scores.to_csv(Path(save_path, "scores.csv"), index=False)

    # plot results
    # sns.set_theme(style="whitegrid")
    # sns.set_context("paper")
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # sns.boxplot(data=scores, x="Protein", y="MAE", ax=ax[0])
    # sns.boxplot(data=scores, x="Protein", y="RMSE", ax=ax[1])
