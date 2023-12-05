from pathlib import Path
import os
from typing import List
import pandas as pd

if __name__ == '__main__':
    # load all files in data/bxs
    data_path = Path("data", "bxs")
    data_frames: [pd.DataFrame] = []
    # iterate over all files excluding subfolders
    loaded_files = 0
    for file in os.listdir(data_path):
        if Path(file).suffix != ".csv":
            continue
        loaded_files += 1
        # load file
        df = pd.read_csv(Path(data_path, file))
        # append to list
        data_frames.append(df)

    data_frames = pd.concat(data_frames, axis=0)
    # count numbers of rows
    print(f"Number of rows: {len(data_frames)}")
    # calculate mean number of rows per biopsy
    print(f"Mean number of rows per biopsy: {len(data_frames) / loaded_files}")
