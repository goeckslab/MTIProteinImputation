import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']

SAVE_PATH = Path("figures", "supplements", "visualize_original_vs_imputed_cells")

if __name__ == '__main__':

    for patient in PATIENTS:
        print(f"Working on patient {patient}...")
        treatments = [1, 2]

        for treatment in treatments:
            isPre: bool = True if treatment == 1 else False

            biopsy_save_path: Path = Path(SAVE_PATH, patient, "pre_treatment" if isPre else "on_treatment")
            if not biopsy_save_path.exists():
                biopsy_save_path.mkdir(parents=True)

            biopsy_df = pd.read_csv(Path("data", "bxs", f"{patient}_{str(treatment)}.csv"))
            imputed_df = pd.read_csv(Path("results", "imputed_data", "ae", "single", "exp", patient, "0",
                                          "pre_treatment.csv" if isPre else "on_treatment.csv"))


            for protein in SHARED_PROTEINS:
                # Sample data for original and imputed expression
                data = pd.DataFrame({
                    'X': biopsy_df['X_centroid'],
                    'Y': biopsy_df['Y_centroid'],
                    'Original Expression': biopsy_df[protein],
                    'Imputed Expression': imputed_df[protein]
                })

                # min max sclae the original and imputed expression
                data['Original Expression'] = MinMaxScaler().fit_transform(data['Original Expression'].values.reshape(-1, 1))
                data['Imputed Expression'] = MinMaxScaler().fit_transform(data['Imputed Expression'].values.reshape(-1, 1))

                # Compute the global min and max of the protein expression
                vmin = min(data['Original Expression'].min(), data['Imputed Expression'].min())
                vmax = max(data['Original Expression'].max(), data['Imputed Expression'].max())

                # Create subplots
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                # Plot original protein expression
                scatter1 = axes[0].scatter(data['X'], data['Y'], c=data['Original Expression'], cmap='viridis', s=5,
                                           vmin=vmin, vmax=vmax)
                scatter1.set_alpha(0.5)
                axes[0].set_title(f'Original {protein} Expression')
                axes[0].set_xlabel('X Coordinate')
                axes[0].set_ylabel('Y Coordinate')
                axes[0].set_xlim(data["X"].min(), data["X"].max())
                axes[0].set_ylim(data["Y"].min(), data["Y"].max())
                fig.colorbar(scatter1, ax=axes[0], label=f'Protein Expression')

                # Plot imputed protein expression
                scatter2 = axes[1].scatter(data['X'], data['Y'], c=data['Imputed Expression'], cmap='viridis', s=5,
                                           vmin=vmin, vmax=vmax)
                # turn down alpha
                scatter2.set_alpha(0.5)
                axes[1].set_title(f'Imputed {protein} Expression')
                axes[1].set_xlabel('X Coordinate')
                axes[1].set_ylabel('Y Coordinate')
                axes[1].set_xlim(data["X"].min(), data["X"].max())
                axes[1].set_ylim(data["Y"].min(), data["Y"].max())
                fig.colorbar(scatter2, ax=axes[1], label=f'Protein Expression')

                plt.tight_layout()
                plt.savefig(Path(biopsy_save_path, f"{protein}_original_vs_imputed.png"), dpi=300)
                plt.close('all')
