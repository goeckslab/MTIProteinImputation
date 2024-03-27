import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
PATIENTS = ["9_2", "9_3"]

image_folder = Path("figures", "figxxx")

if __name__ == '__main__':

    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    # load scores from results/classifier/exp/patient
    scores = []
    for patient in PATIENTS:
        patient_scores = pd.read_csv(f"results/classifier_multi/exp/{patient}/0/classifier_scores.csv")
        patient_scores["Patient"] = patient
        scores.append(patient_scores)

    scores = pd.concat(scores)

    # pivot scores so that these columns Imputed Score,Original Score,Removed score form one column
    scores = scores.melt(id_vars=["Patient", "Round"],
                         value_vars=["Imputed Score", "Original Score", "Removed Score"], value_name="Score",
                         var_name="Type")

    # plot results as comparison between imputed, original and removed score
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.boxplot(data=scores, x="Round", y="Score", hue="Type")
    plt.title("Classifier scores for imputed, original and removed proteins")
    plt.ylabel("Accuracy")
    plt.xlabel("Round")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig_xxx_2.png"), dpi=300)

