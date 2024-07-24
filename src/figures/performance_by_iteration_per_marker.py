import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

save_folder = Path( "figures", "performance_by_iteration_per_marker")

if __name__ == '__main__':

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", "-p", type=str, required=True, help="The protein to plot")
    args = parser.parse_args()
    protein = args.protein

    df = pd.read_csv(Path("results", "temp_scores", "ae", "single_imputation", "scores.csv"))
    assert not df.empty, "Dataframe is empty"
    df = df[df["Marker"] == protein]
    assert not df.empty, "Dataframe is empty"
    df = df[df["Mode"] == "exp"]
    assert not df.empty, "Dataframe is empty"
    df = df[df["FE"] == 0]
    assert not df.empty, "Dataframe is empty"
    df = df[df["Replace Value"] == "mean"]
    assert not df.empty, "Dataframe is empty"
    df = df[df["Noise"] == 0]
    assert not df.empty, "Dataframe is empty"

    print(df)

    fig = plt.figure(figsize=(10, 5), dpi=150)
    # plot a line plot of the MAE by iteration and use the biopsy as hue
    sns.lineplot(data=df, x="Iteration", y="MAE", hue="Biopsy")
    # show every single iteration on x axis
    plt.xticks(df["Iteration"].unique())
    # move legend outside of plot
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
    plt.title(f"Performance by Iteration for {protein}")
    plt.tight_layout()
    plt.savefig(Path(save_folder, f"{protein}.png"), dpi=150)

    #sns.barplot(data=df, x="Iteration", y="MAE", hue="Biopsy")
    #plt.show()
