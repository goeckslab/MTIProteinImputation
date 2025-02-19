import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
import matplotlib.gridspec as gridspec
import numpy as np

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14',
                  'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK', 'EGFR', 'ER']

image_folder = Path("figures", "fig8")

def create_imputed_vs_original_scores(scores: pd.DataFrame):
    scores = scores.melt(id_vars=["Patient", "Protein"],
                         value_vars=["Imputed Score", "Removed Score", "Original Score"],
                         value_name="Score", var_name="Type")
    scores["Type"] = scores["Type"].replace({
        "Imputed Score": "Imputed Data",
        "Removed Score": "Removed Data",
        "Original Score": "Ground Truth Data"
    })
    scores = scores.sort_values(by=["Protein"])
    imputed_mean = scores[scores['Type'] == 'Imputed Data']['Score'].mean()
    ground_truth_mean = scores[scores['Type'] == 'Ground Truth Data']['Score'].mean()
    improvement = imputed_mean - ground_truth_mean
    print(f"Improvement: {improvement}")
    hue_order = ["Ground Truth Data", "Removed Data", "Imputed Data"]
    ax = sns.boxenplot(data=scores, x="Protein", y="Score", hue="Type",
                       hue_order=hue_order,
                       palette={"Ground Truth Data": "yellow",
                                "Imputed Data": "darkgreen",
                                "Removed Data": "red"})
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14',
             'HER2', 'AR', 'CK17', 'p21', 'Vimentin', 'pERK', 'EGFR', 'ER']
    pairs = [
        (("pRB", "Ground Truth Data"), ("pRB", "Imputed Data")),
        (("CD45", "Ground Truth Data"), ("CD45", "Imputed Data")),
        (("CK19", "Ground Truth Data"), ("CK19", "Imputed Data")),
        (("Ki67", "Ground Truth Data"), ("Ki67", "Imputed Data")),
        (("aSMA", "Ground Truth Data"), ("aSMA", "Imputed Data")),
        (("Ecad", "Ground Truth Data"), ("Ecad", "Imputed Data")),
        (("PR", "Ground Truth Data"), ("PR", "Imputed Data")),
        (("CK14", "Ground Truth Data"), ("CK14", "Imputed Data")),
        (("HER2", "Ground Truth Data"), ("HER2", "Imputed Data")),
        (("AR", "Ground Truth Data"), ("AR", "Imputed Data")),
        (("CK17", "Ground Truth Data"), ("CK17", "Imputed Data")),
        (("p21", "Ground Truth Data"), ("p21", "Imputed Data")),
        (("Vimentin", "Ground Truth Data"), ("Vimentin", "Imputed Data")),
        (("pERK", "Ground Truth Data"), ("pERK", "Imputed Data")),
        (("EGFR", "Ground Truth Data"), ("EGFR", "Imputed Data")),
        (("ER", "Ground Truth Data"), ("ER", "Imputed Data")),
        (("pRB", "Ground Truth Data"), ("pRB", "Removed Data")),
        (("CD45", "Ground Truth Data"), ("CD45", "Removed Data")),
        (("CK19", "Ground Truth Data"), ("CK19", "Removed Data")),
        (("Ki67", "Ground Truth Data"), ("Ki67", "Removed Data")),
        (("aSMA", "Ground Truth Data"), ("aSMA", "Removed Data")),
        (("Ecad", "Ground Truth Data"), ("Ecad", "Removed Data")),
        (("PR", "Ground Truth Data"), ("PR", "Removed Data")),
        (("CK14", "Ground Truth Data"), ("CK14", "Removed Data")),
        (("HER2", "Ground Truth Data"), ("HER2", "Removed Data")),
        (("AR", "Ground Truth Data"), ("AR", "Removed Data")),
        (("CK17", "Ground Truth Data"), ("CK17", "Removed Data")),
        (("p21", "Ground Truth Data"), ("p21", "Removed Data")),
        (("Vimentin", "Ground Truth Data"), ("Vimentin", "Removed Data")),
        (("pERK", "Ground Truth Data"), ("pERK", "Removed Data")),
        (("EGFR", "Ground Truth Data"), ("EGFR", "Removed Data")),
        (("ER", "Ground Truth Data"), ("ER", "Removed Data")),
    ]
    annotator = Annotator(ax, pairs, data=scores, x="Protein", y="Score", order=order, hue="Type",
                          hue_order=hue_order, verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()
    ax.legend(loc='lower center', bbox_to_anchor=(0.72, 0.03), ncol=3, prop={"size": 6})
    ax.set_title('Accuracy score', rotation='vertical', x=-0.06, y=0.25, fontsize=12)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    return ax

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    dpi = 300
    image_folder.mkdir(parents=True, exist_ok=True)

    # Load scores for downstream classifier
    og_vs_imputed_scores = []
    for patient in PATIENTS:
        patient_scores = pd.read_csv(f"results/classifier/downstream_classifier/exp/{patient}/0/classifier_scores.csv")
        patient_scores["Patient"] = patient
        og_vs_imputed_scores.append(patient_scores)
    og_vs_imputed_scores = pd.concat(og_vs_imputed_scores)
    # Repeat scores to mimic sample size (if needed)
    og_vs_imputed_scores = pd.concat([og_vs_imputed_scores] * 30)

    # Load images for panels a and b
    downstream_workflow = plt.imread(Path("figures", "fig8", "downstream.png"))
    b_panel = plt.imread(Path("figures", "fig8", "panel_b.png"))

    # Create figure using constrained_layout to reduce whitespace.
    # Figure size is set to 8″ x 10″, which is within A4 limits.
    fig = plt.figure(figsize=(8, 10), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.3)

    # Panel a: Downstream workflow image
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(downstream_workflow, aspect='auto')
    ax1.axis('off')

    # Panel b: b_panel image
    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(b_panel, aspect='equal')
    ax2.axis('off')

    # Panel c: Accuracy scores plot
    ax3 = fig.add_subplot(gs[2, :])
    ax3 = create_imputed_vs_original_scores(og_vs_imputed_scores)

    # Add panel labels using fig.text() with fontsize 12.
    # Here we align the labels vertically using fixed positions.
    fig.text(0, ax1.get_position().y1 + 0.08, "a", ha='left', va='bottom')
    fig.text(0, ax2.get_position().y1 + 0.08, "b", ha='left', va='bottom')
    fig.text(0, ax3.get_position().y1 + 0.01, "c", ha='left', va='bottom')

    # Save the figure with minimal whitespace.
    fig.savefig(Path(image_folder, "fig8.png"), dpi=dpi, bbox_inches='tight', transparent=False)
    fig.savefig(Path(image_folder, "fig8.eps"), dpi=dpi, bbox_inches='tight', format='eps', transparent=False)