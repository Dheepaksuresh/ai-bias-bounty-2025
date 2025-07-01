import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
data = pd.read_csv('datasets/loan_access_dataset.csv')

# Map 'Loan_Approved' to binary
data['Loan_Approved_Binary'] = data['Loan_Approved'].map({'Approved': 1, 'Denied': 0})

# Features to check for bias
bias_features = ['Gender', 'Race', 'Disability_Status', 'Language_Proficiency', 'Zip_Code_Group']

# Create folder for plots
plot_dir = 'outputs/bias_plots'
os.makedirs(plot_dir, exist_ok=True)

# Function to check bias for each feature
def check_bias_and_plot(feature):
    approval_rates = data.groupby(feature)['Loan_Approved_Binary'].mean()
    print(f"\nBias Check - {feature}:")
    print(approval_rates)

    # Plotting
    plt.figure(figsize=(8, 5))
    approval_rates.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Approval Rate by {feature}')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{feature}_bias.png")
    plt.close()

# Run for each feature
for feature in bias_features:
    check_bias_and_plot(feature)

print("\nAll bias plots saved to 'outputs/bias_plots/' ✅")
