import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

patient = "9_2"
df = pd.read_csv(Path("data", "bxs", f"{patient}_1.csv"))
protein = "pRB"

# visualize the biopsy based on the spatial coord X_centroid and Y_centroid
def select_non_overlapping_tiles(df, num_tiles=20):
    selected_tiles = []
    for _, row in df.iterrows():
        if len(selected_tiles) >= num_tiles:
            break
        overlap = any(is_overlapping(row, selected_tile) for selected_tile in selected_tiles)
        if not overlap:
            selected_tiles.append(row)
    return pd.DataFrame(selected_tiles)

def is_overlapping(tile1, tile2):
    return not (tile1['x_end'] < tile2['x_start'] or tile1['x_start'] > tile2['x_end'] or
                tile1['y_end'] < tile2['y_start'] or tile1['y_start'] > tile2['y_end'])


imputed_predictions = pd.read_csv(
    Path("results", "classifier", "informative_tiles", "exp", patient, "0", "experiment_1", "predictions",
         f"{protein}_original_predictions.csv"))

# get pre treatment predictions
pre_tx_predictions = imputed_predictions[imputed_predictions["Treatment"] == "PRE"]
pre_tx_predictions = pre_tx_predictions[
    ["Treatment", "x_start", "x_end", "y_start", "y_end", "prediction_label"]]

# get post treatment predictions
post_tx_predictions = imputed_predictions[imputed_predictions["Treatment"] == "ON"]
post_tx_predictions = post_tx_predictions[
    ["Treatment", "x_start", "x_end", "y_start", "y_end", "prediction_label"]]

# filter only wrong predictions
wrong_pre_tx_predictions = pre_tx_predictions[
    pre_tx_predictions["prediction_label"] != pre_tx_predictions["Treatment"]]
wrong_post_tx_predictions = post_tx_predictions[
    post_tx_predictions["prediction_label"] != post_tx_predictions["Treatment"]]

# filter only correct predictions
correct_pre_tx_predictions = pre_tx_predictions[
    pre_tx_predictions["prediction_label"] == pre_tx_predictions["Treatment"]]
correct_post_tx_predictions = post_tx_predictions[
    post_tx_predictions["prediction_label"] == post_tx_predictions["Treatment"]]

fig = sns.scatterplot(data=df, x="X_centroid", y="Y_centroid", s=10)
# show only x 7000 to 8500, and y 4000 to 6000
plt.xlim(7000, 8500)
plt.ylim(4000, 6000)
# rename X_centroid and Y_centroid to X and Y
plt.xlabel("X")
plt.ylabel("Y")
count = 0

correct_pre_tx_predictions = correct_pre_tx_predictions[correct_pre_tx_predictions["x_start"] < 8500]
correct_pre_tx_predictions = correct_pre_tx_predictions[correct_pre_tx_predictions["x_start"] > 7000]
correct_pre_tx_predictions = correct_pre_tx_predictions[correct_pre_tx_predictions["y_start"] < 6000]
correct_pre_tx_predictions = correct_pre_tx_predictions[correct_pre_tx_predictions["y_start"] > 4000]

# select 20 non overlapping tiles
tiles = select_non_overlapping_tiles(correct_pre_tx_predictions, num_tiles=10)



for i, row in tiles.iterrows():

    x = row["x_start"]
    y = row["y_start"]

    width = row["x_end"] - row["x_start"]
    height = row["y_end"] - row["y_start"]
    fig.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor='green', lw=2))
plt.show()
