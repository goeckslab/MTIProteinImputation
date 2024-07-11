import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
import matplotlib.gridspec as gridspec
import numpy as np

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

image_folder = Path("figures", "fig7")


# Calculate the IQR for x and y
# Calculate the IQR for a given column
def filter_iqr(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    return lower_bound, upper_bound


def create_predictive_tissue(biopsy: pd.DataFrame, matching_tiles: pd.DataFrame):
    # Calculate IQR bounds for X_centroid and Y_centroid
    x_lower, x_upper = filter_iqr(biopsy['X_centroid'])
    y_lower, y_upper = filter_iqr(biopsy['Y_centroid'])

    # Filter the DataFrame based on the IQR bounds
    filtered_df = biopsy[(biopsy['X_centroid'] >= x_lower) & (biopsy['X_centroid'] <= x_upper) &
                         (biopsy['Y_centroid'] >= y_lower) & (biopsy['Y_centroid'] <= y_upper)]

    ax = sns.scatterplot(data=filtered_df, x="X_centroid", y="Y_centroid")
    for i, row in matching_tiles.iterrows():
        x = row["x_start"]
        y = row["y_start"]
        width = row["x_end"] - row["x_start"]
        height = row["y_end"] - row["y_start"]
        ax.add_patch(plt.Rectangle((x, y), width, height, edgecolor='green', facecolor='none'))

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def create_imputed_vs_original_scores(scores: pd.DataFrame):
    # pivot scores so that these columns Imputed Score,Original Score,Removed score form one column
    scores = scores.melt(id_vars=["Patient", "Protein"],
                         value_vars=["Imputed Score", "Original Score"], value_name="Score",
                         var_name="Type")

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    ax = sns.barplot(data=scores, x="Protein", y="Score", hue="Type", hue_order=["Original Score", "Imputed Score"],
                     palette={"Original Score": "green", "Imputed Score": "darkgreen"})

    ax.set_ylabel("")
    ax.set_xlabel("")
    # log y axis
    ax.set_yscale('linear')
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    pairs = [
        (("pRB", "Original Score"), ("pRB", "Imputed Score")),
        (("CD45", "Original Score"), ("CD45", "Imputed Score")),
        (("CK19", "Original Score"), ("CK19", "Imputed Score")),
        (("Ki67", "Original Score"), ("Ki67", "Imputed Score")),
        (("aSMA", "Original Score"), ("aSMA", "Imputed Score")),
        (("Ecad", "Original Score"), ("Ecad", "Imputed Score")),
        (("PR", "Original Score"), ("PR", "Imputed Score")),
        (("CK14", "Original Score"), ("CK14", "Imputed Score")),
        (("HER2", "Original Score"), ("HER2", "Imputed Score")),
        (("AR", "Original Score"), ("AR", "Imputed Score")),
        (("CK17", "Original Score"), ("CK17", "Imputed Score")),
        (("p21", "Original Score"), ("p21", "Imputed Score")),
        (("Vimentin", "Original Score"), ("Vimentin", "Imputed Score")),
        (("pERK", "Original Score"), ("pERK", "Imputed Score")),
        (("EGFR", "Original Score"), ("EGFR", "Imputed Score")),
        (("ER", "Original Score"), ("ER", "Imputed Score"))
    ]

    annotator = Annotator(ax, pairs, data=scores, x="Protein", y="Score", order=order, hue="Type",
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    # change legend position and and add only 1 row
    ax.legend(prop={"size": 7}, loc='center', bbox_to_anchor=[0.8, 0.95], ncol=2)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


if __name__ == '__main__':
    dpi = 300
    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    # load scores from results/classifier/exp/patient
    og_vs_imputed_scores = []
    for patient in PATIENTS:
        patient_scores = pd.read_csv(f"results/classifier/downstream_classifier/exp/{patient}/0/classifier_scores.csv")
        patient_scores["Patient"] = patient
        og_vs_imputed_scores.append(patient_scores)

    og_vs_imputed_scores = pd.concat(og_vs_imputed_scores)
    downstream_workflow = plt.imread(Path("figures", "fig7", "downstream.png"))

    # Create new figure
    # Create the figure and outer GridSpec
    fig = plt.figure(figsize=(10, 7), dpi=100)
    gspec = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.text(0, 1.15, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title("Downstream Workflow", rotation='vertical', x=-0.05, y=0, fontsize=12)
    ax1.imshow(downstream_workflow, aspect='auto')
    # remove y axis from ax1
    ax1.set_yticks([])
    ax1.set_xticks([])

    # Create a nested GridSpec in the second row of the outer GridSpec
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gspec[1], wspace=0.5)

    # Create a subplot in the first column of the nested GridSpec
    ax4 = fig.add_subplot(inner_gs[0, :2])
    ax4.text(0, 1.15, "b", transform=ax4.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax4.set_title('', rotation='vertical', x=-0.05, y=0.3, fontsize=8)
    ax4 = create_predictive_tissue(pd.read_csv("data/bxs/9_2_1.csv"),
                                   pd.read_csv("results/predictive_tissue/9_2/original_pre_matching_tiles.csv"))

    # Create a subplot in the first column of the nested GridSpec
    ax5 = fig.add_subplot(inner_gs[0, 2:])
    ax5.text(0, 1.15, "c", transform=ax5.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax5.set_title('', rotation='vertical', x=-0.05, y=0.3, fontsize=8)
    ax5 = create_predictive_tissue(pd.read_csv("data/bxs/9_2_2.csv"),
                                   pd.read_csv("results/predictive_tissue/9_2/original_post_matching_tiles.csv"))

    # Create a subplot in the third row of the outer GridSpec
    ax3 = fig.add_subplot(gspec[2])
    ax3.text(0, 1.15, "d", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax3.set_title('EN vs LGBM vs AE MAE', rotation='vertical', x=-0.05, y=0.3, fontsize=8)
    ax3 = create_imputed_vs_original_scores(og_vs_imputed_scores)

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig7.png"), dpi=300)
    plt.savefig(Path(image_folder, "fig7.eps"), dpi=300)
