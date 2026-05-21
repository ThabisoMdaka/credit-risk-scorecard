import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("data/raw/german_credit.csv")

print("\nORIGINAL DATA:")
print(df.head())

# -----------------------------
# DROP UNNECESSARY COLUMN
# -----------------------------

df.drop(columns=['Unnamed: 0'], inplace=True)

print("\nCOLUMN DROPPED:")
print(df.head())

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

# Replace missing categorical values
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

# -----------------------------
# ENCODE CATEGORICAL VARIABLES
# -----------------------------

label_encoder = LabelEncoder()

categorical_columns = [
    'Sex',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Purpose'
]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nENCODED DATA:")
print(df.head())

# -----------------------------
# SAVE CLEANED DATASET
# -----------------------------

df.to_csv("data/processed/cleaned_credit_data.csv", index=False)

print("\nCLEANED DATA SAVED SUCCESSFULLY!")