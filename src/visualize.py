import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load predictions
df = pd.read_csv("predictions.csv")

def compute_group_metrics(df, group_col):
    metrics = []
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]
        tn, fp, fn, tp = confusion_matrix(group_data['actual'], group_data['predicted']).ravel()

        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics.append({
            group_col: group,
            "TPR": TPR,
            "FPR": FPR
        })
    
    return pd.DataFrame(metrics)

def plot_group_metrics(metrics_df, title, file_name):
    metrics_df.set_index(metrics_df.columns[0], inplace=True)
    metrics_df.plot(kind='bar')
    plt.title(title)
    plt.ylabel("Rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Gender Bias Plot
gender_metrics_df = compute_group_metrics(df, 'Gender')
plot_group_metrics(gender_metrics_df, "TPR and FPR by Gender", "gender_group_metrics.png")

# Race Bias Plot
race_metrics_df = compute_group_metrics(df, 'Race')
plot_group_metrics(race_metrics_df, "TPR and FPR by Race", "race_group_metrics.png")

# Disability Bias Plot
disability_metrics_df = compute_group_metrics(df, 'Disability_Status')
plot_group_metrics(disability_metrics_df, "TPR and FPR by Disability Status", "disability_group_metrics.png")

print("✅ Fairness visualizations saved.")
