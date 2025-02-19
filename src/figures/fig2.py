import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
from statannotations.Annotator import Annotator
import matplotlib.image as mpimg

warnings.simplefilter(action='ignore', category=FutureWarning)

image_folder = Path("figures", "fig2")
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
PROTEINS_OF_INTEREST = ["aSMA", "CD45", "CK19", "CK14", "CK17"]

phenotype_folder = Path("results", "phenotypes")


# Function for creating the bar plot for Null vs EN models (Panel a)
def create_bar_plot_null_model(data: pd.DataFrame, metric: str, ax=None) -> plt.Axes:
    hue = "Model"
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, hue_order=["Null", "EN"],
                       palette={"EN": "lightblue", "Null": "red"}, ax=ax, showfliers=True)

    # Scale between 0 and 1
    data[metric] = MinMaxScaler().fit_transform(data[metric].values.reshape(-1, 1))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)

    ax.set_xticklabels(
        ['Mean\nof all\nproteins' if x.get_text() == 'Mean' else x.get_text() for x in ax.get_xticklabels()]
    )

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Statistical annotations
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
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order,
                          hue=hue, hue_order=["Null", "EN"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


# Function for creating the bar plot for EN vs LGBM models (Panel b)
def create_bar_plot_en_vs_lgbm(data: pd.DataFrame, metric: str, ax=None) -> plt.Axes:
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue="Network", hue_order=["EN", "LGBM"],
                       palette={"EN": "lightblue", "LGBM": "orange"}, ax=ax)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(bbox_to_anchor=[0.6, 0.85], loc='center', ncol=2)
    ax.set_xticklabels(
        ['Mean\nof all\nproteins' if x.get_text() == 'Mean' else x.get_text() for x in ax.get_xticklabels()]
    )

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Statistical annotations
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
    annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order,
                          hue="Network", hue_order=["EN", "LGBM"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


# === Main Script ===
if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Data Loading and Processing (unchanged)
    null_model_scores = pd.read_csv("results/scores/null_model/scores.csv")
    null_model_scores = null_model_scores.rename(columns={"Protein": "Marker"})
    null_model_scores["MAE"] = (null_model_scores["MAE"] - null_model_scores["MAE"].min()) / (
        null_model_scores["MAE"].max() - null_model_scores["MAE"].min())

    lgbm_scores = pd.read_csv(Path("results", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "AP"]
    lgbm_scores.sort_values(by=["Marker"], inplace=True)
    lgbm_mean = lgbm_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean = lgbm_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    lgbm_mean["Marker"] = "Mean"
    lgbm_mean["FE"] = 0
    lgbm_mean["HP"] = 1
    lgbm_mean["Network"] = "LGBM"
    lgbm_scores = pd.concat([lgbm_scores, lgbm_mean], ignore_index=True)

    en_scores = pd.read_csv(Path("results", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})
    en_scores = en_scores[en_scores["Mode"] == "AP"]
    en_scores.sort_values(by=["Marker"], inplace=True)
    en_mean = en_scores.groupby(["Marker", "Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean = en_mean.groupby(["Mode", "Biopsy"]).mean(numeric_only=True).reset_index()
    en_mean["Marker"] = "Mean"
    en_mean["FE"] = 0
    en_mean["HP"] = 0
    en_mean["Network"] = "EN"
    en_scores = pd.concat([en_scores, en_mean], ignore_index=True)

    null_mean = null_model_scores.groupby(["Model", "Biopsy", "Marker"]).mean(numeric_only=True).reset_index()
    null_mean["Marker"] = "Mean"
    null_mean["FE"] = 0
    null_mean["HP"] = 0
    null_model_scores = pd.concat([null_model_scores, null_mean], ignore_index=True)

    combined_en_lgbm_scores = pd.concat([en_scores, lgbm_scores])

    # Create the figure and sub-grids for panels a, b, c, d using a 4-row gridspec
    fig = plt.figure(figsize=(8, 11), dpi=150)
    gs = fig.add_gridspec(4, 1, hspace=0.8)  # 4 rows, one for each panel

    # Panel a (label "a"): Null vs EN MAE (occupies row 0)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title('Null & EN MAE', rotation='vertical', x=-0.07, y=0)
    ax_a = create_bar_plot_null_model(data=null_model_scores, metric="MAE", ax=ax_a)

    # Panel b (label "b"): EN vs LGBM MAE (occupies row 1)
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_title('EN & LGBM MAE', rotation='vertical', x=-0.07, y=-0.2)
    ax_b = create_bar_plot_en_vs_lgbm(data=combined_en_lgbm_scores, metric="MAE", ax=ax_b)

    # Panel c (label "c"): Vimentin images (occupies row 2)
    sub_gs_c = gs[2, 0].subgridspec(1, 3, wspace=0.3)
    ax_c_left = None
    for i, img_path in enumerate(["figures/fig2/Vimentin_Galaxy.png",
                                  "figures/fig2/Vimentin_Original.png",
                                  "figures/fig2/Vimentin_Imputed.png"]):
        ax = fig.add_subplot(sub_gs_c[0, i])
        img = mpimg.imread(img_path)
        img = np.clip(img * 2.5, 0, 1)
        ax.imshow(img, aspect='auto')
        if i == 0:
            ax.set_title("In Situ")
            ax_c_left = ax  # for panel label "c"
        elif i == 1:
            ax.set_title("Original")
        elif i == 2:
            ax.set_title("Imputed")
        ax.axis('off')
    fig.text(0.535, 0.3, "Vimentin", rotation='horizontal', va='center', ha='right')

    # Panel d (label "d"): PR images (occupies row 3)
    sub_gs_d = gs[3, 0].subgridspec(1, 4, width_ratios=[2, 2, 2, 0.5], wspace=0.3)
    ax_d_left = None
    for i, img_path in enumerate(["figures/fig2/PR_Galaxy.png", "figures/fig2/PR_Original.png",
                                  "figures/fig2/PR_Imputed.png", "figures/fig2/heatmap.png"]):
        ax = fig.add_subplot(sub_gs_d[0, i])
        img = mpimg.imread(img_path)
        img = np.clip(img * 1.5, 0, 1)
        ax.imshow(img, aspect='auto')
        if i == 0:
            ax.set_title("In Situ")
            ax_d_left = ax  # for panel label "d"
        elif i == 1:
            ax.set_title("Original")
        elif i == 2:
            ax.set_title("Imputed")
        ax.axis('off')
    fig.text(0.515, 0.27, "PR", rotation='horizontal',va='center', ha='right')

    plt.box(False)
    plt.tight_layout()

    # Add panel labels using fixed x coordinates
    label_x_left = 0.06
    for label, ax in zip(['a', 'b', 'c'], [ax_a, ax_b, ax_c_left]):
        pos = ax.get_position()
        fig.text(label_x_left, pos.y1 + 0.01, label, ha='left', va='bottom')

    label_x_right = 0.06
    pos = ax_d_left.get_position()
    fig.text(label_x_right, pos.y1 + 0.01, "d", ha='left', va='bottom')

    plt.savefig(Path(image_folder, "fig2.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(image_folder, "fig2.eps"), dpi=300, bbox_inches='tight', format='eps')
    sys.exit()