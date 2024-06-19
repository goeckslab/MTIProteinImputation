import os
from pathlib import Path
import pandas as pd

columns = []

# iterate through data/tma/base and get all column names. Then subset to the shared names
for root, dirs, files in os.walk("data/tma/base"):
    for name in files:
        if Path(name).suffix == ".csv":
            df = pd.read_csv(os.path.join(root, name))
            columns.append(df.columns)


shared_columns = set(columns[0])
for column in columns:
    shared_columns = shared_columns.intersection(column)

shared_columns = list(shared_columns)
# remove all DNA columns
shared_columns = [col for col in shared_columns if "DNA" not in col]
# remove all controls
shared_columns = [col for col in shared_columns if "Control" not in col]
# remove all AF
shared_columns = [col for col in shared_columns if "AF" not in col]
print(shared_columns)

