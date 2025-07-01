import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("datasets/loan_access_dataset.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical columns to numeric (Label Encoding)
df = pd.get_dummies(df, drop_first=True)

# Choose target and features
target = 'LoanApproved_Yes' if 'LoanApproved_Yes' in df.columns else df.columns[-1]  # fallback
X = df.drop(target, axis=1)
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Bias detection (example: Gender_Male)
bias_column = "Gender_Male"
if bias_column in df.columns:
    X_test_df = X_test.copy()
    X_test_df[bias_column] = X_test[bias_column]
    X_test_df["prediction"] = y_pred
    group0 = X_test_df[X_test_df[bias_column] == 0]["prediction"]
    group1 = X_test_df[X_test_df[bias_column] == 1]["prediction"]
    print("\nBias Check - Gender:")
    print("Approval rate for Female (0):", group0.mean())
    print("Approval rate for Male (1):", group1.mean())
else:
    print("\nBias Check column 'Gender_Male' not found in dataset.")

# Save fairness report
with open("fairness_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(classification_report(y_test, y_pred))
    if bias_column in df.columns:
        f.write("\nBias Report (Gender):\n")
        f.write(f"Female approval rate: {group0.mean()}\n")
        f.write(f"Male approval rate: {group1.mean()}\n")
