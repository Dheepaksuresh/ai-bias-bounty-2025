import pandas as pd

df = pd.read_csv("datasets/loan_access_dataset.csv")
print("Column names in the dataset:")
print(df.columns)
print("\nFirst few rows:")
print(df.head())
