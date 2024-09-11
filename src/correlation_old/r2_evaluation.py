import pandas as pd
from sklearn.metrics import r2_score

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':

    original_data = {}
    imputed_data = {}
    for biopsy in BIOPSIES:
        patient = '_'.join(biopsy.split('_')[:2])
        pre: bool = True if "1" == biopsy.split('_')[2] else False

        if biopsy == "9_14_2" or biopsy == "9_15_2":
            assert pre == False, "Patient 9_14_2 and 9_15_2 are post biopsies"

        original = pd.read_csv(f"data/bxs/{biopsy}.csv")
        if pre:
            imputed = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/pre_treatment.csv")
        else:
            imputed = pd.read_csv(f"results/imputed_data/ae/single/exp/{patient}/0/on_treatment.csv")

        original = original[SHARED_MARKERS]
        imputed = imputed[SHARED_MARKERS]

        original_data[biopsy] = original
        imputed_data[biopsy] = imputed


    # for each marker in each biopsy calcualte r2 between the original and imputed data
    r2_scores = []
    for biopsy in BIOPSIES:
        original = original_data[biopsy]
        imputed = imputed_data[biopsy]
        for marker in SHARED_MARKERS:
            r2 = r2_score(original[marker], imputed[marker])
            r2_scores.append({"Biopsy": biopsy, "Marker": marker, "R2": r2})


    r2_scores = pd.DataFrame(r2_scores)
    print(r2_scores)

    # plot
    import seaborn as sns
    from pathlib import Path
    import matplotlib.pyplot as plt
    #image_folder = Path("figures", "r2_evaluation")
    #if not image_folder.exists():
    #    image_folder.mkdir(parents=True)

    hue = "Biopsy"
    x = "Marker"
    ax = sns.barplot(data=r2_scores, x=x, y="R2", hue=hue)
    ax.set_ylabel("R2")
    ax.set_xlabel("Marker")
    ax.set_title("R2 scores for each marker in each biopsy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


