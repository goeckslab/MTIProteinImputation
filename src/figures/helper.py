import pandas as pd

def extract_boxplot_statistics_no_hue(data, metric, group_by="Marker"):
    """
    Extracts summary statistics for each marker, rounding values to 3 decimal places.
    """
    summary_stats = []

    for marker, group in data.groupby(group_by):
        mean_val = round(group[metric].mean(), 3)
        median_val = round(group[metric].median(), 3)
        sem_val = round(group[metric].sem(), 3)
        std_val = round(group[metric].std(), 3)
        min_val = round(group[metric].min(), 3)
        max_val = round(group[metric].max(), 3)
        q1 = round(group[metric].quantile(0.25), 3)
        q3 = round(group[metric].quantile(0.75), 3)

        summary_stats.append({
            "Marker": marker,
            "Mean": mean_val,
            "Median": median_val,
            "SEM": sem_val,
            "SD": std_val,
            "Min": min_val,
            "Max": max_val,
            "Q1 (25%)": q1,
            "Q3 (75%)": q3
        })

    return pd.DataFrame(summary_stats)



def extract_boxplot_statistics(data, metric, group_by="Marker", hue="Model"):
    """
    Extracts summary statistics for each marker and model/network, rounding values to 3 decimal places.
    """
    summary_stats = []

    for (marker, model), group in data.groupby([group_by, hue]):
        mean_val = round(group[metric].mean(), 3)
        median_val = round(group[metric].median(), 3)
        sem_val = round(group[metric].sem(), 3)
        std_val = round(group[metric].std(), 3)
        min_val = round(group[metric].min(), 3)
        max_val = round(group[metric].max(), 3)
        q1 = round(group[metric].quantile(0.25), 3)
        q3 = round(group[metric].quantile(0.75), 3)

        summary_stats.append({
            "Marker": marker,
            "Model": model,
            "Mean": mean_val,
            "Median": median_val,
            "SEM": sem_val,
            "SD": std_val,
            "Min": min_val,
            "Max": max_val,
            "Q1 (25%)": q1,
            "Q3 (75%)": q3
        })

    return pd.DataFrame(summary_stats)