
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

predictions = []
# iterate thorugh results/classifier/informative_tiles
for sub_dir in Path("results", "classifier", "informative_tiles", "exp").iterdir():
    if sub_dir.is_dir():
        for sub_dir in Path(sub_dir).iterdir():
            if not sub_dir.is_dir():
                continue

            for sub_dir in Path(sub_dir).iterdir():
                if not sub_dir.is_dir():
                    continue

                for sub_dir in Path(sub_dir).iterdir():
                    if not sub_dir.is_dir():
                        continue

                    for file in Path(sub_dir).iterdir():
                        if "predictions" == file.parts[-2]:
                            print(file)
                            #load file
                            data = pd.read_csv(file)
                            data["Patient"] = file.parts[-5]
                            data["Marker"] = file.parts[-1].split("_")[0]
                            data["Type"] = file.parts[-1].split("_")[1]
                            data["Accuracy"] = accuracy_score(data["Treatment"], data["prediction_label"])
                            predictions.append(data)

predictions = pd.concat(predictions)

# plot bar plots to show performance per marker and type
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
sns.barplot(data=predictions, x="Marker", y="Accuracy", hue="Type", ax=ax)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Marker")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




