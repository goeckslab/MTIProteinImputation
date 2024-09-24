import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

save_folder = Path("figures", "supplements", "phenotypes")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Define the color mapping for your phenotypes
    color_mapping = {
        "Immune": "yellow",
        "Luminal": "blue",
        "Basal": "red",
        "Unknown": "gray",
        "Stroma (aSMA+)": "green"
    }

    phenotype_data = []

    for biopsy in BIOPSIES:
        df = pd.read_csv(Path("results", "phenotypes", f"{biopsy}_phenotype.csv"))
        df["Biopsy"] = biopsy
        phenotype_data.append(df)

    phenotype_data = pd.concat(phenotype_data)

    # drop imageid column
    phenotype_data = phenotype_data.drop("imageid", axis=1)
    # drop unnamed column
    phenotype_data = phenotype_data.drop("Unnamed: 0", axis=1)
    phenotype_data = phenotype_data.set_index("Biopsy")

    # remove all unknown phenotypes
    # phenotype_data = phenotype_data[phenotype_data["phenotype"] != "Unknown"]

    colors = [color_mapping.get(phenotype, "gray") for phenotype in phenotype_data["phenotype"].value_counts().index]  # Default to gray if not mapped

    # plot the distribution of phenotypes as pieplot
    fig = plt.figure(figsize=(10, 10))
    plt.pie(phenotype_data["phenotype"].value_counts(), labels=phenotype_data["phenotype"].value_counts().index,
            autopct='%1.1f%%', colors=colors)

    plt.tight_layout()
    plt.savefig(Path(save_folder, "phenotype_distribution.png"), dpi=150)

    # Plot the distribution of phenotypes for each biopsy
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for i, biopsy in enumerate(BIOPSIES):
        data = phenotype_data.loc[biopsy]
        data = data["phenotype"].value_counts()

        # Get colors based on phenotype categories
        colors = [color_mapping.get(phenotype, "gray") for phenotype in data.index]  # Default to gray if not mapped

        axs[i % 2, i // 2].pie(data, labels=data.index, autopct='%1.1f%%', colors=colors)
        axs[i % 2, i // 2].set_title(f"Biopsy: {' '.join(biopsy.split('_'))}")

    plt.tight_layout()
    plt.savefig(Path(save_folder, "phenotype_distribution_per_biopsy.png"), dpi=150)
