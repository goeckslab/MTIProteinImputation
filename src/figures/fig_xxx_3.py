import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
#PATIENTS = ["9_2", "9_3"]

image_folder = Path("figures", "figxxx")

if __name__ == '__main__':

    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    # load scores from results/classifier/exp/patient
    scores = []
    for patient in PATIENTS:
        try:
            patient_scores = pd.read_csv(f"results/classifier/new_downstream/{patient}_ae_results.csv")
            patient_scores["Patient"] = patient
            scores.append(patient_scores)
        except:
            continue

    scores = pd.concat(scores)

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    # select only rows where the score is greater than 0.6
    scores = scores[scores["Accuracy"] > 0.6]
    # select only the patients that are left after filtering
    PATIENTS = scores["Patient"].unique()

    # remove patients that have less than 2 proteins
    scores = scores.groupby("Patient").filter(lambda x: len(x) > 1)



    # plot results as comparison between imputed, original and removed score
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.boxplot(data=scores, x="Protein", y="Accuracy", hue="Imputed")
    plt.title("Classifier scores for imputed, original and removed proteins")
    plt.ylabel("Accuracy")
    plt.xlabel("Protein")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    #plt.savefig(Path(image_folder, "fig_xxx.png"), dpi=300)
