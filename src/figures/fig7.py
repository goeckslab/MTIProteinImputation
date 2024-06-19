import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
# PATIENTS = ["9_2", "9_3"]

image_folder = Path("figures", "fig7")


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: []) -> plt.Figure:
    hue = "Model"
    x = "Marker"
    ax = sns.boxenplot(data=data, x=x, y=metric, hue=hue,
                       palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey", "AE M": "darkgrey"})

    # plt.title(title)
    # remove y axis label
    ax.set_ylabel("")
    ax.set_xlabel("")
    # plt.legend(loc='upper center')
    ax.set_ylim([0, 1])

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # remove legend from fig
    ax.legend(prop={"size": 7}, loc='center', bbox_to_anchor=[0.3, 0.7])
    # plt.legend().set_visible(False)

    pairs = []
    for marker in data["Marker"].unique():
        pairs.append(((marker, "EN"), (marker, "AE")))
        # pairs.append(((micron, "LGBM"), (micron, "AE M")))
        # pairs.append(((micron, "AE"), (micron, "AE M")))

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, hue=hue, verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    return ax


def create_imputed_vs_original_scores(scores: pd.DataFrame):
    # pivot scores so that these columns Imputed Score,Original Score,Removed score form one column
    scores = scores.melt(id_vars=["Patient", "Protein"],
                         value_vars=["Imputed Score", "Original Score"], value_name="Score",
                         var_name="Type")

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    ax = sns.boxplot(data=scores, x="Protein", y="Score", hue="Type", hue_order=["Original Score", "Imputed Score"],
                     palette={"Original Score": "green", "Imputed Score": "darkgreen"})

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
    ax.legend(prop={"size": 7}, loc='center', bbox_to_anchor=[0.3, 0.7], ncol=2)

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

    ae_scores = pd.read_csv(Path("results", "tma", "ae_scores.csv"))
    en_scores = pd.read_csv(Path("results", "tma", "en_scores.csv"))

    # select Biopsy, Patient, Marker, MAE columns
    ae_scores = ae_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    en_scores = en_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]

    network_scores = pd.concat([ae_scores, en_scores])
    # select AR, CD45, CK14, CK17, CK18, EGFR, ER, ECA, er2, PR, , Vimentin, p21, pERk, pRB
    network_scores = network_scores[network_scores["Marker"].isin(
        ["AR", "CD45", "CK14", "CK17", "CK19", "EGFR", "ER", "Ecad", "HER2", "PR", "Ki67", "aSMA", "Vimentin",
         "p21", "pERK", "pRB"])]
    print(network_scores)

    # Create new figure
    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    gspec = fig.add_gridspec(2, 1)

    ax1 = fig.add_subplot(gspec[0, :])
    ax1.set_title('AE S 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.05, y=0, fontsize=8)
    ax1 = create_imputed_vs_original_scores(og_vs_imputed_scores)

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('AE S 0 µm', fontsize=8)
    ax2 = create_boxen_plot(network_scores, "MAE", [0, 1])
    # set ax2 title
    ax2.set_title("AE vs. EN", fontsize=8)

    # set colormap

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig7.png"), dpi=300)
