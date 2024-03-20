import argparse, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

save_path: Path = Path("results", "classifier")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', "-r", type=int, default=0, choices=[0, 15, 30, 60, 90, 120])
    parser.add_argument("--patient", "-p", type=str, required=True, help="Patient data to use")
    parser.add_argument("--mode", "-m", type=str, required=True, choices=["ip", "exp"], default="ip")

    args = parser.parse_args()

    radius: int = args.radius
    patient: str = args.patient
    mode: str = args.mode

    print(f"Using radius: {radius} µm")
    print(f"Using patient: {patient}")
    print(f"Using mode: {mode}")

    save_path = Path(save_path, mode, patient, str(radius))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load patient data from the ae single imputation folder
    pre_treatment_path = Path("data", "ae_reconstructed_data", mode, patient, f"{radius}", "pre_treatment.csv")
    on_treatment_path = Path("data", "ae_reconstructed_data", mode, patient, f"{radius}", "on_treatment.csv")

    if not pre_treatment_path.exists() or not on_treatment_path.exists():
        print("Pre or on treatment file does not exist.")
        sys.exit(1)

    # load files
    pre_treatment = pd.read_csv(pre_treatment_path)
    on_treatment = pd.read_csv(on_treatment_path)

    # combine data
    combined_data = pd.concat([pre_treatment, on_treatment])

    imputed_proteins = combined_data[SHARED_MARKERS]

    # load original data
    original_data_pre = pd.read_csv(Path("data", "bxs", f"{patient}_1.csv"))
    original_data_on = pd.read_csv(Path("data", "bxs", f"{patient}_2.csv"))
    # add treatment column
    original_data_pre["Treatment"] = "PRE"
    original_data_on["Treatment"] = "ON"

    original_data = pd.concat([original_data_pre, original_data_on])

    # get the proteins
    original_proteins = original_data[SHARED_MARKERS].copy()
    original_treatment = original_data["Treatment"].copy()

    scores = []
    for target_protein in imputed_proteins:
        print(f"Working on protein {target_protein}...")

        original_proteins[target_protein] = imputed_proteins[target_protein].values

        # create train test split
        X_train, X_test, y_train, y_test = train_test_split(original_proteins, original_treatment,
                                                            test_size=0.2, random_state=42)

        # scale the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # train classifier on masked data using random forest using sklearn
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        imputed_score = model.score(X_test, y_test)

        print(f"Score for protein {target_protein} using imputed data: {imputed_score}")

        # create train test split using original data, but remove target protein
        X_train, X_test, y_train, y_test = train_test_split(original_proteins.drop(columns=[target_protein]),
                                                            original_treatment,
                                                            test_size=0.2, random_state=42)

        # shape of X_train and X_test should be one less that the original_proteins
        assert X_train.shape[1] == original_proteins.shape[1] - 1, "X_train shape is not correct"

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # train classifier on original data using random forest using sklearn
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        original_score = model.score(X_test, y_test)

        print(f"Score for protein {target_protein} using original data: {original_score}")

        # save the scores
        scores.append({"Protein": target_protein, "Imputed Score": imputed_score, "Original Score": original_score})

    scores = pd.DataFrame(scores)
    scores.to_csv(Path(save_path, "classifier_scores.csv"), index=False)
