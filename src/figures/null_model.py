import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
PATIENTS = ["9_2", "9_3"]

image_folder = Path("figures", "null_model")

if __name__ == '__main__':

    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    scores = pd.read_csv(f"results/scores/null_model/null_model.csv")

    # sort by proteins
    scores = scores.sort_values(by=["Protein"])

    # plot results as comparison between imputed, original and removed score
    fig = plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.boxplot(data=scores, x="Protein", y="MAE")
    plt.title("MAE by protein for null model")
    plt.ylabel("MAE")
    plt.xlabel("Protein")
    plt.xticks(rotation=45)
    # clip between 0 and 1.5
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig_xxx.png"), dpi=300)
