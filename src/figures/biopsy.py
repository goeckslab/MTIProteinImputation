import pandas as pd
from pathlib import Path


df = pd.read_csv(Path("data", "bxs", "9_2_1.csv"))

# visualize the biopsy based on the spatial coord X_centroid and Y_centroid

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=df, x="X_centroid", y="Y_centroid", s=10)
plt.show()
