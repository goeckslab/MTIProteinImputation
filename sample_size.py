import pandas as pd
from pathlib import Path

biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]

sum = 0
for biopsy in biopsies:
    df = pd.read_csv(f"data/bxs/{biopsy}.csv")
    sum += len(df)

print("Total number of samples:", sum)


tile_sum = 0

for file in Path("results", "classifier", "informative_tiles", "exp", "9_2", "0", "experiment_1", "data").iterdir():
    if file.is_dir():
        continue

    if file.suffix == ".csv":
        print(f"Loading file {file}")
        df = pd.read_csv(file)
        tile_sum += len(df)

print("Total number of tiles:", tile_sum)