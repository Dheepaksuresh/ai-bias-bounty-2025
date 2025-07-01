import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("datasets/loan_access_dataset.csv")

# Encode target label
df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': True, 'Denied': False})

# Encode sensitive attributes
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Drop unnecessary columns
drop_cols = ['ID', 'Loan_Approved', 'Zip_Code_Group']
X = df.drop(columns=drop_cols)
y = df['Loan_Approved']

# Convert categorical features using one-hot encoding
X = pd.get_dummies(X)

# 🔧 Fill missing values with column means (you can also use median or mode)
X = X.fillna(X.mean(numeric_only=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save predictions to CSV
results_df = X_test.copy()
results_df['prediction'] = y_pred
results_df['actual'] = y_test.values
results_df.to_csv("predictions.csv", index=False)

# Bias check for gender
X_test_with_gender = X_test.copy()
X_test_with_gender['Gender'] = df.loc[X_test.index, 'Gender']
X_test_with_gender['prediction'] = y_pred

female_approval_rate = X_test_with_gender[X_test_with_gender['Gender'] == 0]['prediction'].mean()
male_approval_rate = X_test_with_gender[X_test_with_gender['Gender'] == 1]['prediction'].mean()

print("\nBias Check - Gender:")
print(f"Approval rate for Female (0): {female_approval_rate}")
print(f"Approval rate for Male (1): {male_approval_rate}")
