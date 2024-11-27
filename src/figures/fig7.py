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


# Function to check overlap between two tiles
def is_overlapping(tile1, tile2):
    return not (tile1['x_end'] < tile2['x_start'] or tile1['x_start'] > tile2['x_end'] or
                tile1['y_end'] < tile2['y_start'] or tile1['y_start'] > tile2['y_end'])


# Function to select non-overlapping tiles
def select_non_overlapping_tiles(df, num_tiles=20):
    selected_tiles = []
    for _, row in df.iterrows():
        if len(selected_tiles) >= num_tiles:
            break
        overlap = any(is_overlapping(row, selected_tile) for selected_tile in selected_tiles)
        if not overlap:
            selected_tiles.append(row)
    return pd.DataFrame(selected_tiles)


def create_predictive_tissue(biopsy: pd.DataFrame, matching_tiles: pd.DataFrame, x_start, y_start, x_end, y_end):
    # Calculate IQR bounds for X_centroid and Y_centroid
    # x_lower, x_upper = filter_iqr(biopsy['X_centroid'])
    # y_lower, y_upper = filter_iqr(biopsy['Y_centroid'])
    x_lower, x_upper = x_start, x_end
    y_lower, y_upper = y_start, y_end

    # Filter the DataFrame based on the IQR bounds
    filtered_df = biopsy[(biopsy['X_centroid'] >= x_lower) & (biopsy['X_centroid'] <= x_upper) &
                         (biopsy['Y_centroid'] >= y_lower) & (biopsy['Y_centroid'] <= y_upper)]

    ax = sns.scatterplot(data=filtered_df, x="X_centroid", y="Y_centroid")

    # filter matching tiles
    matching_tiles = matching_tiles[(matching_tiles["x_start"] >= x_lower) & (matching_tiles["x_end"] <= x_upper) &
                                    (matching_tiles["y_start"] >= y_lower) & (matching_tiles["y_end"] <= y_upper)]

    # only keep 20 distinct non overlapping tiles
    matching_tiles = select_non_overlapping_tiles(matching_tiles, num_tiles=25)

    for i, row in matching_tiles.iterrows():
        x = row["x_start"]
        y = row["y_start"]
        width = row["x_end"] - row["x_start"]
        height = row["y_end"] - row["y_start"]
        ax.add_patch(plt.Rectangle((x, y), width, height, edgecolor='green', facecolor='none', linewidth=2.5))

    # rename x and y axis
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def create_imputed_vs_original_scores(scores: pd.DataFrame):
    # pivot scores so that these columns Imputed Score,Original Score,Removed score form one column
    scores = scores.melt(id_vars=["Patient", "Protein"],
                         value_vars=["Imputed Score", "Removed Score", "Original Score"], value_name="Score",
                         var_name="Type")

    # rename Imputed Score to Imputed Data, Removed Score to Removed Data, Original Score to Original Data
    scores["Type"] = scores["Type"].replace({"Imputed Score": "Imputed Data", "Removed Score": "Removed Data",
                                             "Original Score": "Ground Truth Data"})

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    # calculate improvement for imputed vs ground truth data and calculate overall mean
    print(f"Improvement: {scores[scores['Type'] == 'Imputed Data']['Score'].mean() - scores[scores['Type'] == 'Ground Truth Data']['Score'].mean()}")

    hue_order = ["Ground Truth Data", "Removed Data", "Imputed Data"]
    ax = sns.barplot(data=scores, x="Protein", y="Score", hue="Type",
                     hue_order=hue_order,
                     palette={"Ground Truth Data": "yellow", "Imputed Data": "darkgreen", "Removed Data": "red"})

    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_ylim(0, 1)
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
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
                          verbose=1, hue_order=hue_order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()


    # add legend
    ax.legend(loc='center', bbox_to_anchor=[0.5, 0.95], ncol=3, prop={"size": 6})
    ax.set_title('Accuracy score', rotation='vertical', x=-0.06, y=0.25, fontsize=10)

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
    og_vs_imputed_scores = pd.concat([og_vs_imputed_scores] * 30)
    downstream_workflow = plt.imread(Path("figures", "fig7", "downstream.png"))
    b_panel = plt.imread(Path("figures", "fig7", "panel_b.png"))

    # Create a new figure and outer GridSpec
    fig = plt.figure(figsize=(12, 10), dpi=300)  # Adjusted figure size for better clarity
    gspec = gridspec.GridSpec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.4)  # Adjusted height ratios and spacing

    # First row subplot
    ax1 = fig.add_subplot(gspec[0, :])
    ax1.text(-0.08, 1.05, "a", transform=ax1.transAxes,  # Aligned to the left
             fontsize=12, fontweight='bold', va='top', ha='left')
    ax1.imshow(downstream_workflow, aspect='auto')
    ax1.axis('off')

    # Second row subplot for Panel B
    ax3 = fig.add_subplot(gspec[1, :])
    ax3.text(-0.17, 1.05, "b", transform=ax3.transAxes,  # Aligned to the left
             fontsize=12, fontweight='bold', va='top', ha='left')
    ax3.imshow(b_panel, aspect='equal')  # Set aspect to 'equal' to avoid stretching
    ax3.axis('off')

    # Third row subplot for accuracy scores
    ax3 = fig.add_subplot(gspec[2])
    ax3.text(-0.08, 1.05, "c", transform=ax3.transAxes,  # Aligned to the left
             fontsize=12, fontweight='bold', va='top', ha='left')
    ax3 = create_imputed_vs_original_scores(og_vs_imputed_scores)
    plt.tight_layout()

    # Save the figure
    plt.savefig(Path(image_folder, "fig7.png"), dpi=300)
    plt.savefig(Path(image_folder, "fig7.eps"), dpi=300)