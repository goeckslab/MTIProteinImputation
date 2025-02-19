import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2',
                  'AR', 'CK17', 'p21', 'Vimentin', 'pERK', 'EGFR', 'ER']

image_folder = Path("figures", "fig4")


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: list) -> plt.Figure:
    hue = "Model"
    x = "Marker"
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue, hue_order=["EN", "LGBM", "AE"],
                       palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey", "AE M": "darkgrey"})

    # Remove axis labels
    ax.set_ylabel("")
    ax.set_xlabel("")
    # Reduce tick label font size
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove plot spines for a cleaner look
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Add statistical annotations
    pairs = []
    for marker in data["Marker"].unique():
        pairs.append(((marker, "LGBM"), (marker, "AE")))
        pairs.append(((marker, "EN"), (marker, "LGBM")))
        pairs.append(((marker, "EN"), (marker, "AE")))

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, hue=hue, verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    # Adjust legend: one row, placed in the upper right corner
    ax.legend(loc='upper center', ncol=2)

    return ax


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    dpi = 300

    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    # Load and process scores for each model
    ae_scores = pd.read_csv(Path("results", "tma", "ae_scores.csv"))
    ae_scores = ae_scores[ae_scores["Marker"].isin(SHARED_MARKERS)]
    ae_scores = ae_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    # Scale MAE between 0 and 1
    ae_scores["MAE"] = (ae_scores["MAE"] - ae_scores["MAE"].min()) / (ae_scores["MAE"].max() - ae_scores["MAE"].min())

    en_scores = pd.read_csv(Path("results", "tma", "en_scores.csv"))
    en_scores = en_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    en_scores["MAE"] = (en_scores["MAE"] - en_scores["MAE"].min()) / (en_scores["MAE"].max() - en_scores["MAE"].min())

    lgbm_scores = pd.read_csv(Path("results", "tma", "lgbm_scores.csv"))
    lgbm_scores = lgbm_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    lgbm_scores["MAE"] = (lgbm_scores["MAE"] - lgbm_scores["MAE"].min()) / (lgbm_scores["MAE"].max() - lgbm_scores["MAE"].min())

    # Concatenate all network scores
    network_scores = pd.concat([ae_scores, en_scores, lgbm_scores])
    network_scores = network_scores[network_scores["Marker"].isin(SHARED_MARKERS)]

    # (Optional) Assert that AE and EN share the same markers
    assert set(ae_scores["Marker"].unique()) == set(en_scores["Marker"].unique())

    # Create a figure that is smaller than or equal to A4.
    # Here we choose 7" x 10", which is within the A4 bounds of 8.27" x 11.69"
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    gspec = fig.add_gridspec(1, 1)

    ax1 = fig.add_subplot(gspec[:, :])
    ax1.set_title('EN vs LGBM vs AE MAE', rotation='vertical', x=-0.05, y=0.3, fontsize=12)
    ax1 = create_boxen_plot(network_scores, "MAE", [0, 1])

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig4.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig4.eps"), dpi=dpi, bbox_inches='tight', format='eps')