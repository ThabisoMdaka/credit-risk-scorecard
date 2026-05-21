import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/german_credit.csv")

# -----------------------------
# BASIC INFORMATION
# -----------------------------

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATASET INFO:")
print(df.info())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# -----------------------------
# VISUAL ANALYSIS
# -----------------------------

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Save plot
plt.savefig("outputs/plots/age_distribution.png")

# Show plot
plt.show()

# Credit Amount Distribution
plt.figure(figsize=(8,5))
plt.hist(df['Credit amount'], bins=20)
plt.title('Credit Amount Distribution')
plt.xlabel('Credit Amount')
plt.ylabel('Frequency')

# Save plot
plt.savefig("outputs/plots/credit_amount_distribution.png")

# Show plot
plt.show()
# -----------------------------
# PURPOSE DISTRIBUTION
# -----------------------------

plt.figure(figsize=(10,5))
df['Purpose'].value_counts().plot(kind='bar')

plt.title('Loan Purpose Distribution')
plt.xlabel('Purpose')
plt.ylabel('Count')

plt.xticks(rotation=45)

plt.savefig("outputs/plots/purpose_distribution.png")

plt.show()

# -----------------------------
# HOUSING DISTRIBUTION
# -----------------------------

plt.figure(figsize=(6,5))
df['Housing'].value_counts().plot(kind='bar')

plt.title('Housing Distribution')
plt.xlabel('Housing Type')
plt.ylabel('Count')

plt.savefig("outputs/plots/housing_distribution.png")

plt.show()

# -----------------------------
# CREDIT AMOUNT BOXPLOT
# -----------------------------

plt.figure(figsize=(8,5))
plt.boxplot(df['Credit amount'], vert=False)

plt.title('Credit Amount Boxplot')

plt.savefig("outputs/plots/credit_amount_boxplot.png")

plt.show()
