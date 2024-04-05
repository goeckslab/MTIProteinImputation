import pandas as pd
from pathlib import Path

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER', "Core"]

if __name__ == '__main__':
    df = pd.read_csv(Path("data", "tma", "tma_single_cell.tsv"), sep="\t")
    df = df[SHARED_MARKERS]
    df.to_csv(Path("data", "tma", "tma_cleaned.tsv"), sep="\t", index=False)

