import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
# PATIENTS = ["9_2", "9_3"]

image_folder = Path("figures", "figxxx")

if __name__ == '__main__':

    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    # load scores from results/classifier/exp/patient
    scores = []
    for patient in PATIENTS:
        patient_scores = pd.read_csv(f"results/classifier/pycaret_tiles/exp/{patient}/0/classifier_scores.csv")
        patient_scores["Patient"] = patient
        scores.append(patient_scores)

    scores = pd.concat(scores)

    # pivot scores so that these columns Imputed Score,Original Score,Removed score form one column
    scores = scores.melt(id_vars=["Patient", "Protein"],
                         value_vars=["Imputed Score", "Original Score"], value_name="Score",
                         var_name="Type")

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])


    # plot results as comparison between imputed, original and removed score
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    ax = sns.boxplot(data=scores, x="Protein", y="Score", hue="Type", hue_order=["Original Score", "Imputed Score"])

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
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    plt.title("Classifier scores for imputed and original protein expressions")
    plt.ylabel("Accuracy")
    plt.xlabel("Protein")
    plt.xticks(rotation=45)

    # set colormap

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig_xxx.png"), dpi=300)
