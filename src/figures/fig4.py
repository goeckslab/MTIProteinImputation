import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator

PATIENTS = ["9_2", "9_3", "9_14", "9_15"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

image_folder = Path("figures", "fig4")


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: []) -> plt.Figure:
    hue = "Model"
    x = "Marker"
    ax = sns.barplot(data=data, x=x, y=metric, hue=hue, hue_order=["EN", "LGBM", "AE"],
                       palette={"EN": "lightblue", "LGBM": "orange", "AE": "grey", "AE M": "darkgrey"})

    # plt.title(title)
    # remove y axis label
    ax.set_ylabel("")
    ax.set_xlabel("")
    # plt.legend(loc='upper center')
    #ax.set_yscale('log', base=10)
    # set ylim
    #ax.set_ylim(ylim)

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    pairs = []
    for marker in data["Marker"].unique():
        pairs.append(((marker, "LGBM"), (marker, "AE")))
        pairs.append(((marker, "EN"), (marker, "LGBM")))
        pairs.append(((marker, "EN"), (marker, "AE")))

    annotator = Annotator(ax, pairs, data=data, x=x, y=metric, hue=hue, verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                        comparisons_correction="Benjamini-Hochberg")
    annotator.apply_and_annotate()

    # change legend position and and add only 1 row
    ax.legend(prop={"size": 7}, loc='center right', bbox_to_anchor=[1, 0.95], ncol=3)

    return ax


if __name__ == '__main__':
    dpi = 300
    if not image_folder.exists():
        image_folder.mkdir(parents=True)

    ae_scores = pd.read_csv(Path("results", "tma", "ae_scores.csv"))
    ae_scores = ae_scores[ae_scores["Marker"].isin(SHARED_MARKERS)]
    ae_scores = ae_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    # scale scores between 0 and 1
    ae_scores["MAE"] = (ae_scores["MAE"] - ae_scores["MAE"].min()) / (ae_scores["MAE"].max() - ae_scores["MAE"].min())

    en_scores = pd.read_csv(Path("results", "tma", "en_scores.csv"))
    # select Biopsy, Patient, Marker, MAE columns
    en_scores = en_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    # scale scores between 0 and 1
    en_scores["MAE"] = (en_scores["MAE"] - en_scores["MAE"].min()) / (en_scores["MAE"].max() - en_scores["MAE"].min())


    lgbm_scores = pd.read_csv(Path("results", "tma", "lgbm_scores.csv"))
    lgbm_scores = lgbm_scores[["Biopsy", "Patient", "Marker", "MAE", "Model"]]
    # scale scores between 0 and 1
    lgbm_scores["MAE"] = (lgbm_scores["MAE"] - lgbm_scores["MAE"].min()) / (lgbm_scores["MAE"].max() - lgbm_scores["MAE"].min())

    network_scores = pd.concat([ae_scores, en_scores, lgbm_scores])
    # select
    network_scores = network_scores[network_scores["Marker"].isin(SHARED_MARKERS)]
    # print(network_scores)
    # create set of ae and en scores

    # assert that both EN and AE have the same unique markers
    assert set(ae_scores["Marker"].unique()) == set(en_scores["Marker"].unique())

    # Create new figure
    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    gspec = fig.add_gridspec(1, 1)

    ax1 = fig.add_subplot(gspec[:, :])
    #ax1.text(0, 1.15, "b", transform=ax1.transAxes,
    #         fontsize=12, fontweight='bold', va='top', ha='right')
    ax1.set_title('EN vs LGBM vs AE MAE', rotation='vertical', x=-0.05, y=0.3, fontsize=8)
    ax1 = create_boxen_plot(network_scores, "MAE", [0, 1])
    # set ax2 tit

    plt.tight_layout()
    plt.savefig(Path(image_folder, "fig4.png"), dpi=300)
    plt.savefig(Path(image_folder, "fig4.eps"), dpi=300)
