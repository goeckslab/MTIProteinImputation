import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

load_folder = Path("results", "phenotypes")


if __name__ == '__main__':
    scores = pd.read_csv(Path(load_folder, "silhouette.csv"))

    # plot bar plot Original vs Imputed
    fig =