import random
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
import os, argparse, logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import List, Dict
from tqdm import tqdm
from patient_mapping import patient_mapping

SHARED_MARKERS = ['AR', 'Ki67', 'CK14', 'aSMA', 'ER', 'HER2', 'EGFR', 'p21', 'Vimentin',
                  'Ecad', 'CK17', 'pERK', 'PR', 'pRB', 'CK19']


def clean_column_names(df: pd.DataFrame):
    if "ERK-1" in df.columns:
        # Rename ERK to pERK
        df = df.rename(columns={"ERK-1": "pERK"})

    if "E-cadherin" in df.columns:
        df = df.rename(columns={"E-cadherin": "Ecad"})

    if "Rb" in df.columns:
        df = df.rename(columns={"Rb": "pRB"})

    return df


class AutoEncoder(Model):
    def __init__(self, input_dim: int, latent_dim: int):
        super(AutoEncoder, self).__init__()
        self._latent_dim = latent_dim
        self.encoder: Model = Sequential([
            Dense(input_dim, activation='relu', name="input"),
            Dense(self._latent_dim, name="latent_space"),
        ])
        self.decoder: Model = Sequential([
            Dense(input_dim, activation="relu", name="output"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def save_scores(save_folder: str, file_name: str, new_scores: pd.DataFrame):
    if Path(f"{save_folder}/{file_name}").exists():
        temp_df = pd.read_csv(f"{save_folder}/{file_name}")
        new_scores = pd.concat([temp_df, pd.DataFrame(new_scores)])
    new_scores.to_csv(f"{save_folder}/{file_name}", index=False)


def impute_markers(scores: List, test_data: pd.DataFrame, all_predictions: Dict,
                   mode: str, spatial_radius: int, experiment_id: int, save_folder: str, file_name: str,
                   replace_value: str, iterations: int, store_predictions: bool, subset: int):
    try:
        for marker in SHARED_MARKERS:
            if verbose:
                print(f"Imputing marker {marker}")
            # copy the test data
            input_data = test_data.copy()
            if replace_value == "zero":
                input_data[marker] = 0
                if int(spatial_radius) > 0:
                    input_data[f"{marker}_mean"] = 0
            elif replace_value == "mean":
                mean = input_data[marker].mean()
                input_data[marker] = mean
                if int(spatial_radius) > 0:
                    input_data[f"{marker}_mean"] = mean

            marker_prediction = input_data.copy()
            for iteration in range(iterations):

                predicted_intensities = ae.decoder.predict(ae.encoder.predict(marker_prediction))

                predicted_intensities = pd.DataFrame(data=predicted_intensities, columns=test_data.columns)
                if store_predictions:
                    all_predictions[iteration][marker] = predicted_intensities[marker].values

                imputed_marker = predicted_intensities[marker].values
                marker_prediction[marker] = imputed_marker

                scores.append({
                    "Marker": marker,
                    "Biopsy": core,
                    "MAE": mean_absolute_error(marker_prediction[marker], test_data[marker]),
                    "RMSE": root_mean_squared_error(marker_prediction[marker], test_data[marker]),
                    "HP": 0,
                    "Mode": mode,
                    "Imputation": 1,
                    "Iteration": iteration,
                    "FE": spatial_radius,
                    "Experiment": int(f"{experiment_id}{subset}"),
                    "Network": "AE",
                    "Noise": 0,
                    "Patient": patient_mapping[core],
                    "Replace Value": replace_value
                })

                if iteration % 20 == 0:
                    if verbose:
                        print("Performing temp save...")
                    save_scores(save_folder=save_folder, file_name=file_name, new_scores=pd.DataFrame(scores))
                    scores = []

        if len(scores) > 0:
            if verbose:
                print("Saving remaining scores...")
            save_scores(save_folder=save_folder, file_name=file_name, new_scores=pd.DataFrame(scores))
            scores = []
        return all_predictions

    except KeyboardInterrupt as ex:
        print("Keyboard interrupt detected.")
        print("Saving scores...")
        if len(scores) > 0:
            save_scores(new_scores=pd.DataFrame(scores), save_folder=save_folder, file_name=file_name)
        raise

    except Exception as ex:
        logging.error(ex)
        logging.error("Test truth:")
        logging.error(test_data)
        logging.error("Predictions:")
        logging.error(predictions)
        logging.error(mode)
        logging.error(spatial_radius)
        logging.error(experiment_id)
        logging.error(replace_value)

        raise


def create_results_folder(core: str, spatial_radius: str) -> [Path, int]:
    save_folder = Path(f"src/tma/ae")
    save_folder = Path(save_folder, core)
    save_folder = Path(save_folder, spatial_radius)

    experiment_id = 0

    base_path = Path(save_folder, "experiment_run")
    save_path = Path(str(base_path) + "_" + str(experiment_id))
    while Path(save_path).exists():
        save_path = Path(str(base_path) + "_" + str(experiment_id))
        experiment_id += 1

    created: bool = False
    if not save_path.exists():
        while not created:
            try:
                save_path.mkdir(parents=True)
                created = True
            except:
                experiment_id += 1
                save_path = Path(str(base_path) + "_" + str(experiment_id))

    return save_path, experiment_id - 1 if experiment_id != 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--core", type=str, required=True,
                        help="The core to impute")
    parser.add_argument("-i", "--iterations", action="store", default=10, type=int)
    # add verbose bool flag
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity", default=False)
    args = parser.parse_args()

    iterations: int = args.iterations
    verbose = args.verbose

    # Load test data
    core = args.core
    patient: str = patient_mapping[core]
    scores_file_name = "scores.csv"

    save_folder, experiment_id = create_results_folder(core=core, spatial_radius="0")

    # Load noisy train data
    train_data = []
    base_path = Path("data", "tma", "base")
    # print(f"Base path: {base_path}")
    for file in os.listdir(base_path):
        file_name = Path(file).stem
        file_patient = patient_mapping[file_name]

        if file.endswith(".csv") and patient not in file_patient:
            if verbose:
                print("Loading train file: " + file)
            data = pd.read_csv(Path(base_path, file))
            data = clean_column_names(data)
            train_data.append(data)
        else:
            if verbose:
                print("Skipping file: " + file)

    train_data = pd.concat(train_data)

    if verbose:
        print("Selecting marker")
    train_data = train_data[SHARED_MARKERS].copy()
    assert train_data.shape[1] == len(SHARED_MARKERS), "Train data not complete"

    # Load test data
    test_data = pd.read_csv(Path(base_path, f'{core}.csv'))
    test_data = clean_column_names(test_data)

    test_data = test_data[SHARED_MARKERS].copy()
    assert test_data.shape[1] == len(SHARED_MARKERS), "Test data not complete"

    # Scale data
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(train_data + 1)),
                              columns=train_data.columns)
    test_data = pd.DataFrame(min_max_scaler.fit_transform(np.log10(test_data + 1)),
                             columns=test_data.columns)

    # Split noisy train data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    scores = []
    predictions = {}
    for i in range(iterations):
        predictions[i] = pd.DataFrame(columns=SHARED_MARKERS)

    # Create ae
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    ae = AutoEncoder(input_dim=train_data.shape[1], latent_dim=5)
    ae.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    history = ae.fit(train_data, train_data, epochs=100, batch_size=32, shuffle=True,
                     validation_data=(val_data, val_data), callbacks=callbacks, verbose=0)

    for i in tqdm(range(30)):
        # sample new dataset from test_data
        test_data_sample = test_data.sample(frac=0.7, random_state=random.randint(0, 100000), replace=True)

        # Predict
        impute_markers(scores=scores, all_predictions=predictions, mode="exp", spatial_radius=0,
                       experiment_id=experiment_id, replace_value="mean",
                       iterations=iterations,
                       store_predictions=False, test_data=test_data_sample, subset=i, file_name=scores_file_name,
                       save_folder=save_folder)
