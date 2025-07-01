import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv("datasets/loan_access_dataset.csv")

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# List of sensitive features to analyze
sensitive_features = [
    ("Gender", {"Male": 1, "Female": 0}),
    ("Race", None),
    ("Age_Group", None),
    ("Disability_Status", None),
    ("Criminal_Record", None)
]

# Iterate through sensitive features and create approval rate barplots
for feature, mapping in sensitive_features:
    if mapping:
        df[f"{feature}_Label"] = df[feature].map(mapping)
        group_col = f"{feature}_Label"
    else:
        group_col = feature

    approval_rates = df.groupby(group_col)['Loan_Approved'].apply(lambda x: (x == 'Approved').mean())

    # Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=approval_rates.index, y=approval_rates.values)
    plt.title(f"Loan Approval Rate by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Approval Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"plots/{feature.lower()}_bias.png")
    plt.close()

print("✅ All bias visualizations saved in 'plots/' folder")
