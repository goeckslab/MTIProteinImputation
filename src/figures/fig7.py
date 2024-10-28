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
                                             "Original Score": "Original Data"})

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    hue_order = ["Original Data", "Removed Data", "Imputed Data"]
    ax = sns.barplot(data=scores, x="Protein", y="Score", hue="Type",
                     hue_order=hue_order,
                     palette={"Original Data": "yellow", "Imputed Data": "darkgreen", "Removed Data": "red"})

    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_ylim(0, 1)
    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    pairs = [
        (("pRB", "Original Data"), ("pRB", "Imputed Data")),
        (("CD45", "Original Data"), ("CD45", "Imputed Data")),
        (("CK19", "Original Data"), ("CK19", "Imputed Data")),
        (("Ki67", "Original Data"), ("Ki67", "Imputed Data")),
        (("aSMA", "Original Data"), ("aSMA", "Imputed Data")),
        (("Ecad", "Original Data"), ("Ecad", "Imputed Data")),
        (("PR", "Original Data"), ("PR", "Imputed Data")),
        (("CK14", "Original Data"), ("CK14", "Imputed Data")),
        (("HER2", "Original Data"), ("HER2", "Imputed Data")),
        (("AR", "Original Data"), ("AR", "Imputed Data")),
        (("CK17", "Original Data"), ("CK17", "Imputed Data")),
        (("p21", "Original Data"), ("p21", "Imputed Data")),
        (("Vimentin", "Original Data"), ("Vimentin", "Imputed Data")),
        (("pERK", "Original Data"), ("pERK", "Imputed Data")),
        (("EGFR", "Original Data"), ("EGFR", "Imputed Data")),
        (("ER", "Original Data"), ("ER", "Imputed Data")),

        (("pRB", "Original Data"), ("pRB", "Removed Data")),
        (("CD45", "Original Data"), ("CD45", "Removed Data")),
        (("CK19", "Original Data"), ("CK19", "Removed Data")),
        (("Ki67", "Original Data"), ("Ki67", "Removed Data")),
        (("aSMA", "Original Data"), ("aSMA", "Removed Data")),
        (("Ecad", "Original Data"), ("Ecad", "Removed Data")),
        (("PR", "Original Data"), ("PR", "Removed Data")),
        (("CK14", "Original Data"), ("CK14", "Removed Data")),
        (("HER2", "Original Data"), ("HER2", "Removed Data")),
        (("AR", "Original Data"), ("AR", "Removed Data")),
        (("CK17", "Original Data"), ("CK17", "Removed Data")),
        (("p21", "Original Data"), ("p21", "Removed Data")),
        (("Vimentin", "Original Data"), ("Vimentin", "Removed Data")),
        (("pERK", "Original Data"), ("pERK", "Removed Data")),
        (("EGFR", "Original Data"), ("EGFR", "Removed Data")),
        (("ER", "Original Data"), ("ER", "Removed Data")),

    ]

    annotator = Annotator(ax, pairs, data=scores, x="Protein", y="Score", order=order, hue="Type",
                          verbose=1, hue_order=hue_order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    # change legend position and and add only 1 row
    # ax.legend(prop={"size": 6}, loc='center', bbox_to_anchor=[0.82, 0.95], ncol=3)

    # Extract handles and labels from the seaborn plot
    # Get handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()

    # Split the legend into two parts: the first for "Original Score" and "Removed Score", and the second for "Imputed Score"
    first_legend_handles = handles[:2]
    first_legend_labels = labels[:2]
    second_legend_handles = handles[2:]
    second_legend_labels = labels[2:]

    # Add the first legend to the plot (for "Original Score" and "Removed Score")
    first_legend = ax.legend(first_legend_handles, first_legend_labels, loc='center', prop={"size": 6}, ncol=2,
                             bbox_to_anchor=[0.55, 0.95])

    # Add the second legend manually (for "Imputed Score")
    ax.add_artist(first_legend)  # Keep the first legend on the plot
    ax.legend(second_legend_handles, second_legend_labels, loc='center', prop={"size": 6}, ncol=1,
              bbox_to_anchor=[0.92, 0.95])

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

    # Create new figure
    # Create the figure and outer GridSpec
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gspec = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.text(0, 1.15, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    plt.box(False)
    ax1.set_title("Downstream Workflow", rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax1.imshow(downstream_workflow, aspect='auto')
    # remove y axis from ax1
    ax1.set_yticks([])
    ax1.set_xticks([])

    # Create a nested GridSpec in the second row of the outer GridSpec
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gspec[1], wspace=0.5)

    # Create a subplot in the first column of the nested GridSpec
    ax4 = fig.add_subplot(inner_gs[0, :])
    ax4.text(0, 1.15, "b", transform=ax4.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax4.set_title('', rotation='vertical', x=-0.05, y=0.3, fontsize=8)
    ax4.imshow(b_panel, aspect='auto')

    # Create a subplot in the third row of the outer GridSpec
    ax3 = fig.add_subplot(gspec[2])
    ax3.text(0, 1.15, "d", transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')

    ax3.set_title('Original, Removed & Imputed\n Accuracy score', rotation='vertical', x=-0.06, y=-0.1, fontsize=8)
    ax3 = create_imputed_vs_original_scores(og_vs_imputed_scores)

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig7.png"), dpi=300)
    plt.savefig(Path(image_folder, "fig7.eps"), dpi=300)
