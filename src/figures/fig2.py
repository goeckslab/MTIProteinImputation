import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from statannotations.Annotator import Annotator
import matplotlib.colors as mcolors  # Proper import for rgb2hex
import matplotlib.image as mpimg

warnings.simplefilter(action='ignore', category=FutureWarning)

image_folder = Path("figures", "fig2")
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_PROTEINS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                   'pERK', 'EGFR', 'ER']
PROTEINS_OF_INTEREST = ["aSMA", "CD45", "CK19", "CK14", "CK17"]
phenotype_folder = Path("results", "phenotypes")


# Function for creating the bar plot for Null vs EN models
def create_bar_plot_null_model(data: pd.DataFrame, metric: str, ax=None) -> plt.Axes:
    hue = "Model"
    ax = sns.barplot(data=data, x="Marker", y=metric, hue=hue, hue_order=["Null", "EN"],
                     palette={"EN": "lightblue", "Null": "red"}, ax=ax)

    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Protein")
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    for label in ax.get_xticklabels():
        if label.get_text() == 'Mean':
            label.set_fontstyle('italic')
            label.set_fontweight('bold')

    ax.set_xticklabels(
        ['Mean\nof all\nproteins' if x.get_text() == 'Mean' else x.get_text() for x in ax.get_xticklabels()])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add statistical annotations
    pairs = [
        (("pRB", "Null"), ("pRB", "EN")),
        (("CD45", "Null"), ("CD45", "EN")),
        (("CK19", "Null"), ("CK19", "EN")),
        (("Ki67", "Null"), ("Ki67", "EN")),
        (("aSMA", "Null"), ("aSMA", "EN")),
        (("Ecad", "Null"), ("Ecad", "EN")),
        (("PR", "Null"), ("PR", "EN")),
        (("CK14", "Null"), ("CK14", "EN")),
        (("HER2", "Null"), ("HER2", "EN")),
        (("AR", "Null"), ("AR", "EN")),
        (("CK17", "Null"), ("CK17", "EN")),
        (("p21", "Null"), ("p21", "EN")),
        (("Vimentin", "Null"), ("Vimentin", "EN")),
        (("pERK", "Null"), ("pERK", "EN")),
        (("EGFR", "Null"), ("EGFR", "EN")),
        (("ER", "Null"), ("ER", "EN")),
        (("Mean", "Null"), ("Mean", "EN"))
    ]

    order = SHARED_PROTEINS + ["Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=["Null", "EN"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


# Function for creating the bar plot for EN vs LGBM models
def create_bar_plot_en_vs_lgbm(data: pd.DataFrame, metric: str, ax=None) -> plt.Axes:
    ax = sns.barplot(data=data, x="Marker", y=metric, hue="Network", hue_order=["EN", "LGBM"],
                     palette={"EN": "lightblue", "LGBM": "orange"}, ax=ax)

    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Protein")
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    for label in ax.get_xticklabels():
        if label.get_text() == 'Mean':
            label.set_fontstyle('italic')
            label.set_fontweight('bold')

    ax.set_xticklabels(
        ['Mean\nof all\nproteins' if x.get_text() == 'Mean' else x.get_text() for x in ax.get_xticklabels()])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add statistical annotations
    pairs = [
        (("pRB", "EN"), ("pRB", "LGBM")),
        (("CD45", "EN"), ("CD45", "LGBM")),
        (("CK19", "EN"), ("CK19", "LGBM")),
        (("Ki67", "EN"), ("Ki67", "LGBM")),
        (("aSMA", "EN"), ("aSMA", "LGBM")),
        (("Ecad", "EN"), ("Ecad", "LGBM")),
        (("PR", "EN"), ("PR", "LGBM")),
        (("CK14", "EN"), ("CK14", "LGBM")),
        (("HER2", "EN"), ("HER2", "LGBM")),
        (("AR", "EN"), ("AR", "LGBM")),
        (("CK17", "EN"), ("CK17", "LGBM")),
        (("p21", "EN"), ("p21", "LGBM")),
        (("Vimentin", "EN"), ("Vimentin", "LGBM")),
        (("pERK", "EN"), ("pERK", "LGBM")),
        (("EGFR", "EN"), ("EGFR", "LGBM")),
        (("ER", "EN"), ("ER", "LGBM")),
        (("Mean", "EN"), ("Mean", "LGBM"))
    ]
    order = SHARED_PROTEINS + ["Mean"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue="Network",
                          hue_order=["EN", "LGBM"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


# Function to plot ARI
def plot_ari():
    results = pd.read_csv("results/evaluation/cluster_metrics.csv")
    ax = sns.barplot(data=results, x="Marker", y="ARI", palette="tab20")
    ax.set_ylabel("Expression ARI Score")
    ax.set_xlabel("Protein")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # **Step 3: Extract Color Assignments**
    color_assignments = []

    # Get the list of marker labels from the x-axis tick labels
    markers = [tick.get_text() for tick in ax.get_xticklabels()]

    # Iterate over each patch (bar) and corresponding marker
    for i, (patch, marker) in enumerate(zip(ax.patches, markers)):
        # Check if patch is a Rectangle (bar)
        if not isinstance(patch, plt.Rectangle):
            print(f"Skipping non-rectangle patch at index {i}")
            continue

        # Get the height of the bar (ARI score)
        height = patch.get_height()

        # Get the face color of the bar (RGBA tuple)
        facecolor = patch.get_facecolor()

        # Convert RGBA to Hex for easier interpretation
        facecolor_hex = mcolors.rgb2hex(facecolor)

        # Append the information as a dictionary
        color_assignments.append({
            'Marker': marker,
            'ARI': height,
            'Color_RGBA': facecolor,
            'Color_Hex': facecolor_hex
        })

    colors = {}
    for assignment in color_assignments:
        if assignment['Marker'] in PROTEINS_OF_INTEREST:
            colors[assignment['Marker']] = assignment['Color_RGBA']

    return ax, colors


def plot_phenotype_ari(ari_scores: pd.DataFrame, color_palette: dict):
    ax = sns.barplot(data=ari_scores, x="Protein", y="Score", palette=color_palette)
    ax.set_ylabel("Phenotype ARI Score")
    ax.set_xlabel("Protein")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_ylim(0, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def plot_phenotype_jaccard(jaccard_scores: pd.DataFrame, color_palette: dict):
    hue_order = ["Original CV Score", "Imputed CV Score"]
    ax = sns.barplot(data=jaccard_scores, x="Protein", y="Score", hue_order=hue_order, palette=color_palette)
    ax.set_ylabel("Phenotype Jaccard Score")
    ax.set_xlabel("Protein")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def plot_silhouette():
    results = pd.read_csv("results/evaluation/cluster_metrics.csv")
    # create 30 replicates of the data
    results = pd.concat([results] * 30, ignore_index=True)

    melt = results.melt(id_vars=["Biopsy", "Marker"],
                        value_vars=["Silhouette Original", "Silhouette Imputed"],
                        var_name="Silhouette Type", value_name="Score")

    # rename value Silhouette Original to Original and Silhoutte Imputed to Imputed
    melt["Silhouette Type"] = melt["Silhouette Type"].replace(
        {"Silhouette Original": "Original", "Silhouette Imputed": "Imputed"})
    ax = sns.barplot(data=melt, x="Marker", y="Score", hue="Silhouette Type")
    ax.set_ylabel("Expression Silhouette Score")
    ax.set_xlabel("Protein")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # adjust legend
    ax.legend(bbox_to_anchor=[0.4, 0.97], loc='center', ncol=2, fontsize=8)

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # Add statistical annotations
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


if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Load and process Null model scores
    null_model_scores = pd.read_csv(f"results/scores/null_model/scores.csv")
    null_model_scores = null_model_scores.rename(columns={"Protein": "Marker"})
    null_model_scores["MAE"] = (null_model_scores["MAE"] - null_model_scores["MAE"].min()) / (
            null_model_scores["MAE"].max() - null_model_scores["MAE"].min())

    # Load LGBM model scores
    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "AP"]
    lgbm_scores.sort_values(by=["Marker"], inplace=True)

    # Calculate the mean performance for each marker and mode
    lgbm_mean = lgbm_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean = lgbm_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean["Marker"] = "Mean"
    lgbm_mean["FE"] = 0
    lgbm_mean["HP"] = 1
    lgbm_mean["Network"] = "LGBM"
    lgbm_scores = pd.concat([lgbm_scores, lgbm_mean], ignore_index=True)

    # Load EN model scores
    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})
    en_scores = en_scores[en_scores["Mode"] == "AP"]
    en_scores.sort_values(by=["Marker"], inplace=True)

    # Calculate the mean performance for EN
    en_mean = en_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean = en_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean["Marker"] = "Mean"
    en_mean["FE"] = 0
    en_mean["HP"] = 0
    en_mean["Network"] = "EN"
    en_scores = pd.concat([en_scores, en_mean], ignore_index=True)

    # Process Null model for adding Mean row
    null_mean = null_model_scores.groupby(["Model", "Biopsy", "Marker"]).mean(numeric_only=True).reset_index()
    null_mean["Marker"] = "Mean"
    null_mean["FE"] = 0
    null_mean["HP"] = 0
    null_model_scores = pd.concat([null_model_scores, null_mean], ignore_index=True)

    # Combine EN and LGBM scores
    combined_en_lgbm_scores = pd.concat([en_scores, lgbm_scores])

    phenotype_scores = pd.read_csv(Path(phenotype_folder, "patient_metrics.csv"))
    # sort the dataframe by the protein
    phenotype_scores = phenotype_scores.sort_values(by="Protein")

    ari_scores = pd.melt(phenotype_scores, id_vars=["Biopsy", "Protein"],
                         value_vars=["ARI Score"],
                         var_name="ARI", value_name="Score")
    # sort the dataframe by the protein
    ari_scores = ari_scores.sort_values(by="Protein")

    jaccard_scores = pd.melt(phenotype_scores, id_vars=["Biopsy", "Protein"],
                             value_vars=["Jaccard"],
                             var_name="Jaccard", value_name="Score")
    # sort the dataframe by the protein
    jaccard_scores = jaccard_scores.sort_values(by="Protein")

    # Create the figure with a grid specification
    fig = plt.figure(figsize=(17, 17), dpi=150)
    gspec = fig.add_gridspec(12, 4)  # Updated to 12 rows to accommodate the new plot

    # First bar plot (Null & EN)
    ax1 = fig.add_subplot(gspec[0:2, :])
    ax1.text(-0.02, 1.2, "a", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax1.set_title('Null & EN MAE', rotation='vertical', x=-0.04, y=0.25, fontsize=12)
    ax1 = create_bar_plot_null_model(data=null_model_scores, metric="MAE", ax=ax1)

    # Second bar plot (EN & LGBM)
    ax2 = fig.add_subplot(gspec[2:4, :])
    ax2.text(-0.02, 1.2, "b", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax2.set_title('EN & LGBM MAE', rotation='vertical', x=-0.04, y=0.25, fontsize=12)
    ax2 = create_bar_plot_en_vs_lgbm(data=combined_en_lgbm_scores, metric="MAE", ax=ax2)

    # New sub-grid: In Vivo & Original & Imputed Expression (three-panel images)
    sub_gspec_c = gspec[4:6, :2].subgridspec(1, 3)  # 1 row, 3 columns within specified main grid area for label "c"
    for i, img_path in enumerate(["figures/fig2/Vimentin_Galaxy.png", "figures/fig2/Vimentin_Original.png",
                                  "figures/fig2/Vimentin_Imputed.png"]):
        ax = fig.add_subplot(sub_gspec_c[0, i])
        img = mpimg.imread(img_path)
        ax.imshow(img, aspect='auto')
        if i == 0:
            ax.set_title("In Situ", fontsize=8)
        elif i == 1:
            ax.set_title("Original", fontsize=8)
        elif i == 2:
            ax.set_title("Imputed", fontsize=8)
        ax.axis('off')  # Turn off axis for a clean display

    # Add label "c" to the figure, aligned to the sub-grid
    fig.text(0.015, 0.64, "c", fontsize=12, fontweight='bold', va='top', ha='right')

    # Define the main grid with a narrower fourth column
    sub_gspec_d = gspec[4:6, 2:].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.2])  # Adjust width ratios as needed
    for i, img_path in enumerate(
            ["figures/fig2/PR_Galaxy.png", "figures/fig2/PR_Original.png", "figures/fig2/PR_Imputed.png",
             "figures/fig2/heatmap.png"]):
        ax = fig.add_subplot(sub_gspec_d[0, i])
        img = mpimg.imread(img_path)
        ax.imshow(img, aspect='auto')
        # Set title for each image
        if i == 0:
            ax.set_title("In Situ", fontsize=8)
        elif i == 1:
            ax.set_title("Original", fontsize=8)
        elif i == 2:
            ax.set_title("Imputed", fontsize=8)
        ax.axis('off')  # Turn off axis for a clean display

    # Add label "d" to the figure, aligned to the sub-grid
    fig.text(0.515, 0.64, "d", fontsize=12, fontweight='bold', va='top', ha='right')

    # Third bar plot (ARI)
    ax4 = fig.add_subplot(gspec[6:8, :2])
    ax4.text(-0.05, 1.1, "e", transform=ax4.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax4, color_palette = plot_ari()

    # Fourth bar plot (Silhouette)
    ax5 = fig.add_subplot(gspec[6:8, 2:])
    ax5.text(-0.065, 1.1, "f", transform=ax5.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax5 = plot_silhouette()

    # Fifth bar plot (Phenotype ARI)
    ax6 = fig.add_subplot(gspec[8:10, :2])
    ax6.text(-0.05, 1.1, "g", transform=ax6.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax6 = plot_phenotype_ari(ari_scores, color_palette)

    # Sixth bar plot (Phenotype Jaccard)
    ax7 = fig.add_subplot(gspec[8:10, 2:])
    ax7.text(-0.065, 1.1, "h", transform=ax7.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')
    ax7 = plot_phenotype_jaccard(jaccard_scores, color_palette)

    plt.box(False)

    # Ensure tight layout to avoid overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(Path(image_folder, "fig2.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig2.eps"), dpi=300, bbox_inches='tight', format='eps')

    sys.exit()
