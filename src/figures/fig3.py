import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from statannotations.Annotator import Annotator
import matplotlib.ticker as mticker
from helper import extract_boxplot_statistics_no_hue, extract_boxplot_statistics

warnings.simplefilter(action='ignore', category=FutureWarning)

# Directories and constants
image_folder = Path("figures", "fig3")
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14',
                   'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK', 'EGFR', 'ER']
SHARED_PROTEINS_COLOR_PALETTE = {
    'pRB': '#1f77b4',
    'CD45': '#ff7f0e',
    'CK19': '#2ca02c',
    'Ki67': '#d62728',
    'aSMA': '#9467bd',
    'Ecad': '#8c564b',
    'PR': '#e377c2',
    'CK14': '#7f7f7f',
    'HER2': '#bcbd22',
    'AR': '#17becf',
    'CK17': '#1f77b4',
    'p21': '#ff7f0e',
    'Vimentin': '#2ca02c',
    'pERK': '#d62728',
    'EGFR': '#9467bd',
    'ER': '#8c564b'
}
phenotype_folder = Path("results", "phenotypes")


# --- Panel plotting functions ---
def plot_ari(color_palette: dict):
    results = pd.read_csv("results/evaluation/cluster_metrics.csv")

    stats = extract_boxplot_statistics_no_hue(results, "ARI", group_by="Marker")

    print(f"ARI: {results.groupby('Marker').mean().mean()}")
    print("Boxenplot ARI statistics:")
    print(stats)
    ax = sns.boxenplot(data=results, x="Marker", y="ARI", palette=color_palette)
    ax.set_ylabel("Expression ARI Score")
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.set_xlabel("Protein")
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    return ax


def plot_phenotype_ari(ari_scores: pd.DataFrame, color_palette: dict):
    print(f"Phenotype ARI: {ari_scores.groupby('Protein').mean().mean()}")
    ax = sns.boxenplot(data=ari_scores, x="Protein", y="Score", palette=color_palette)
    ax.set_ylabel("Phenotype ARI Score")
    ax.yaxis.set_label_coords(-0.18, 0.5)
    ax.set_xlabel("Protein")
    ax.set_ylim(0, 1)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    return ax


def plot_phenotype_jaccard(jaccard_scores: pd.DataFrame, color_palette: dict):
    print(f"Phenotype Jaccard: {jaccard_scores.groupby('Protein').mean().mean()}")
    hue_order = ["Original CV Score", "Imputed CV Score"]
    ax = sns.boxenplot(data=jaccard_scores, x="Protein", y="Score", hue_order=hue_order,
                       palette=color_palette)
    ax.set_ylabel("Phenotype Jaccard Score")
    ax.set_xlabel("Protein")
    ax.set_ylim(0, 1)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    return ax


def plot_silhouette():
    results = pd.read_csv("results/evaluation/cluster_metrics.csv")
    print(f"Silhouette: {results.groupby('Marker').mean().mean()}")
    results["Difference"] = results["Silhouette Imputed"] - results["Silhouette Original"]
    print("Silhouette Imputed improvement:")
    print(results.groupby("Marker")["Difference"].mean().mean())

    melt = results.melt(id_vars=["Biopsy", "Marker"],
                        value_vars=["Silhouette Original", "Silhouette Imputed"],
                        var_name="Silhouette Type", value_name="Score")
    melt["Silhouette Type"] = melt["Silhouette Type"].replace(
        {"Silhouette Original": "Original", "Silhouette Imputed": "Imputed"})

    print("Boxenplot Silhouette statistics:")
    stats = extract_boxplot_statistics(melt, metric="Score", group_by="Marker", hue="Silhouette Type")
    print(stats)

    ax = sns.boxenplot(data=melt, x="Marker", y="Score", hue="Silhouette Type",
                       showfliers=False, palette='Greys')
    ax.set_ylabel("Expression Silhouette Score")
    ax.set_xlabel("Protein")
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.legend(bbox_to_anchor=[0.4, 0.97], loc='center', ncol=2, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    pairs = [
        (("pRB", "Original"), ("pRB", "Imputed")),
        (("CD45", "Original"), ("CD45", "Imputed")),
        (("CK19", "Original"), ("CK19", "Imputed")),
        (("Ki67", "Original"), ("Ki67", "Imputed")),
        (("aSMA", "Original"), ("aSMA", "Imputed")),
        (("Ecad", "Original"), ("Ecad", "Imputed")),
        (("PR", "Original"), ("PR", "Imputed")),
        (("CK14", "Original"), ("CK14", "Imputed")),
        (("HER2", "Original"), ("HER2", "Imputed")),
        (("AR", "Original"), ("AR", "Imputed")),
        (("CK17", "Original"), ("CK17", "Imputed")),
        (("p21", "Original"), ("p21", "Imputed")),
        (("Vimentin", "Original"), ("Vimentin", "Imputed")),
        (("pERK", "Original"), ("pERK", "Imputed")),
        (("EGFR", "Original"), ("EGFR", "Imputed")),
        (("ER", "Original"), ("ER", "Imputed")),
    ]
    order = SHARED_PROTEINS
    annotator = Annotator(ax, pairs, data=melt, x="Marker", y="Score", order=order, hue="Silhouette Type",
                          hue_order=["Original", "Imputed"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()
    return ax


# --- Main Script ---
if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    dpi = 150

    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Load and process phenotype scores
    phenotype_scores = pd.read_csv(Path(phenotype_folder, "patient_metrics.csv"))
    phenotype_scores = phenotype_scores.sort_values(by="Protein")
    ari_scores = pd.melt(phenotype_scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["ARI Score"],
                         var_name="ARI", value_name="Score")
    ari_scores = ari_scores.sort_values(by="Protein")
    jaccard_scores = pd.melt(phenotype_scores, id_vars=["Biopsy", "Protein"],
                             value_vars=["Jaccard"],
                             var_name="Jaccard", value_name="Score")
    jaccard_scores = jaccard_scores.sort_values(by="Protein")

    # Create the figure with 3 rows:
    # Row 0: Panel e, Row 1: Panel f, Row 2: Panels g and h (split into 2 columns)
    fig = plt.figure(figsize=(8, 11), dpi=dpi)
    gs_main = fig.add_gridspec(3, 1, hspace=0.6)

    # Panel a: ARI plot (Row 0, full width)
    ax_a = fig.add_subplot(gs_main[0, 0])
    ax_a = plot_ari(SHARED_PROTEINS_COLOR_PALETTE)

    # Panel b: Silhouette plot (Row 1, full width)
    ax_b = fig.add_subplot(gs_main[1, 0])
    ax_b = plot_silhouette()
    ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Panel g & h: Bottom row split into 2 columns
    gs_bottom = gs_main[2, 0].subgridspec(1, 2, wspace=0.3)
    ax_c = fig.add_subplot(gs_bottom[0, 0])
    ax_c = plot_phenotype_ari(ari_scores, SHARED_PROTEINS_COLOR_PALETTE)
    ax_d = fig.add_subplot(gs_bottom[0, 1])
    ax_d = plot_phenotype_jaccard(jaccard_scores, SHARED_PROTEINS_COLOR_PALETTE)

    plt.tight_layout()

    # Add panel labels using fixed positions
    fig.text(0.05, 0.87, "a", ha='left', va='bottom', fontsize=12)
    fig.text(0.05, 0.62, "b", ha='left', va='bottom', fontsize=12)
    fig.text(0.05, 0.30, "c", ha='left', va='bottom', fontsize=12)
    fig.text(0.50, 0.30, "d", ha='left', va='bottom', fontsize=12)

    # Save the figure (ensuring it does not exceed A4 dimensions)
    plt.savefig(Path(image_folder, "fig3.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig3.eps"), dpi=dpi, bbox_inches='tight', format='eps')

    print("Phenotype ARI scores")
    ari_stats = extract_boxplot_statistics_no_hue(data=ari_scores, metric="Score", group_by="Protein")
    print(ari_stats)

    print("Phenotype Jaccard scores")
    jaccard_stats = extract_boxplot_statistics_no_hue(data=jaccard_scores, metric="Score", group_by="Protein")
    print(jaccard_stats)

    sys.exit()
