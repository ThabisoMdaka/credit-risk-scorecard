import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CREATE MODELS FOLDER
# ===============================
os.makedirs('models', exist_ok=True)

print("=" * 55)
print("   CREDIT RISK — Saving Trained Models")
print("=" * 55)

# ===============================
# LOAD & PREPARE DATA
# ===============================
df = pd.read_csv("data/processed/cleaned_credit_data.csv")
df['Risk'] = ((df['Credit amount'] > 4000) &
              (df['Duration'] > 24)).astype(int)

X = df.drop(columns=['Risk'])
y = df['Risk']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y
)

# ===============================
# TRAIN & SAVE ALL MODELS
# ===============================

# 1. Logistic Regression
print("\n--- Training & Saving Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("Saved: models/logistic_regression.pkl")

# 2. Random Forest
print("\n--- Training & Saving Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("Saved: models/random_forest.pkl")

# 3. XGBoost
print("\n--- Training & Saving XGBoost ---")
xgb = XGBClassifier(
    n_estimators=100, random_state=42,
    eval_metric='logloss', verbosity=0
)
xgb.fit(X_train, y_train)
with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb, f)
print("Saved: models/xgboost.pkl")

# 4. Save Scaler
print("\n--- Saving Scaler ---")
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved: models/scaler.pkl")

# 5. Save Feature Names
feature_names = list(X.columns)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("Saved: models/feature_names.pkl")

print("\n" + "="*55)
print("   ALL MODELS SAVED SUCCESSFULLY!")
print("="*55)
print("\nModels directory:")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}')
    print(f"  {f:<35} {size/1024:.1f} KB")
print("="*55)