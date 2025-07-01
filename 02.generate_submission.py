import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("datasets/loan_access_dataset.csv")

# Clean and encode target
df = df.dropna().copy()
df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})

# Drop ID for features
X = df.drop(['ID', 'Loan_Approved'], axis=1)
y = df['Loan_Approved']

# Encode categorical variables
X = pd.get_dummies(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
test_ids = df.iloc[y_test.index]['ID']

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Create submission dataframe with correct label format
submission = pd.DataFrame({
    'ID': test_ids,
    'Loan_Approved': preds  # Column name must match required format
})

# Convert 1 → 'Approved', 0 → 'Denied'
submission['Loan_Approved'] = submission['Loan_Approved'].map({1: 'Approved', 0: 'Denied'})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created in Approved/Denied format.")
