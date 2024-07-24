import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator

# requires imputed scores for AE and null model

image_folder = Path("figures", "null_model")

if __name__ == '__main__':

    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    null_model_scores = pd.read_csv(f"results/scores/null_model/null_model.csv")
    ae_scores = pd.read_csv(f"results/scores/ae/scores.csv")

    # retrieve only scores for exp, radius 0, mean imputation, noise 0
    ae_scores = ae_scores[ae_scores["Mode"] == "EXP"]
    ae_scores = ae_scores[ae_scores["FE"] == 0]
    ae_scores = ae_scores[ae_scores["Replace Value"] == "mean"]
    ae_scores = ae_scores[ae_scores["Noise"] == 0]

    # drop hp, fe, noise, replace value, experiment
    ae_scores = ae_scores.drop(columns=["HP", "FE", "Noise", "Replace Value", "Mode", "Experiment"])
    null_model_scores = null_model_scores.drop(columns=["Iteration"])
    # rename Marker to protein in ae scores
    ae_scores = ae_scores.rename(columns={"Marker": "Protein"})
    # add Network to null MOdel
    null_model_scores["Network"] = "Null Model"

    # merge the two dataframes
    scores = pd.concat([null_model_scores, ae_scores])

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    # plot results as comparison between imputed, original and removed score
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    ax = sns.boxplot(data=scores, x="Protein", y="MAE", hue="Network", showfliers=False)

    order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
             'pERK', 'EGFR', 'ER']
    pairs = [
        (("pRB", "AE"), ("pRB", "Null Model")),
        (("CD45", "AE"), ("CD45", "Null Model")),
        (("CK19", "AE"), ("CK19", "Null Model")),
        (("Ki67", "AE"), ("Ki67", "Null Model")),
        (("aSMA", "AE"), ("aSMA", "Null Model")),
        (("Ecad", "AE"), ("Ecad", "Null Model")),
        (("PR", "AE"), ("PR", "Null Model")),
        (("CK14", "AE"), ("CK14", "Null Model")),
        (("HER2", "AE"), ("HER2", "Null Model")),
        (("AR", "AE"), ("AR", "Null Model")),
        (("CK17", "AE"), ("CK17", "Null Model")),
        (("p21", "AE"), ("p21", "Null Model")),
        (("Vimentin", "AE"), ("Vimentin", "Null Model")),
        (("pERK", "AE"), ("pERK", "Null Model")),
        (("EGFR", "AE"), ("EGFR", "Null Model")),
        (("ER", "AE"), ("ER", "Null Model"))
    ]

    annotator = Annotator(ax, pairs, data=scores, x="Protein", y="MAE", order=order, hue="Network",
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    plt.title("MAE by Protein")
    plt.ylabel("MAE")
    plt.xlabel("Protein")
    plt.xticks(rotation=45)
    # clip between 0 and 1.5
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig_xxx.png"), dpi=300)
