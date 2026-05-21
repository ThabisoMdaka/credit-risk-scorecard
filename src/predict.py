import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CREDIT RISK PREDICTION PIPELINE
# Author: Thabiso Mdaka
# ===============================

def load_models():
    """Load all saved models and scaler"""
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest':       joblib.load('models/random_forest.pkl'),
        'XGBoost':             joblib.load('models/xgboost.pkl')
    }
    scaler        = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return models, scaler, feature_names


def encode_borrower(borrower: dict) -> dict:
    """
    Encode raw borrower input into model-ready format.

    Encoding maps:
    Sex:              male=1, female=0
    Housing:          own=1, free=0, rent=2
    Saving accounts:  little=0, moderate=1,
                      quite rich=2, rich=3, unknown=4
    Checking account: little=0, moderate=1,
                      rich=2, unknown=3
    Purpose:          car=1, furniture=4, radio/TV=5,
                      education=3, business=0,
                      repairs=6, domestic=2, vacation=7
    """
    encoding = {
        'Sex': {'male': 1, 'female': 0},
        'Housing': {'own': 1, 'free': 0, 'rent': 2},
        'Saving accounts': {
            'little': 0, 'moderate': 1,
            'quite rich': 2, 'rich': 3, 'unknown': 4
        },
        'Checking account': {
            'little': 0, 'moderate': 1,
            'rich': 2, 'unknown': 3
        },
        'Purpose': {
            'car': 1, 'furniture/equipment': 4,
            'radio/TV': 5, 'education': 3,
            'business': 0, 'repairs': 6,
            'domestic appliances': 2, 'vacation/others': 7
        }
    }

    encoded = borrower.copy()
    for col, mapping in encoding.items():
        if col in encoded:
            encoded[col] = mapping.get(
                encoded[col], 0
            )
    return encoded


def predict_risk(borrower: dict):
    """
    Predict credit risk for a single borrower.

    Parameters:
    -----------
    borrower : dict
        Dictionary containing borrower features

    Returns:
    --------
    results : dict
        Predictions from all models with
        risk scores and recommendations
    """
    models, scaler, feature_names = load_models()

    # Encode categorical variables
    encoded = encode_borrower(borrower)

    # Create DataFrame in correct feature order
    input_df = pd.DataFrame([encoded])[feature_names]

    # Scale features
    input_scaled = scaler.transform(input_df)

    results = {}
    for model_name, model in models.items():
        proba    = model.predict_proba(input_scaled)[0][1]
        decision = 'HIGH RISK — DECLINE' if proba > 0.5 \
                   else 'LOW RISK — APPROVE'
        confidence = proba * 100 if proba > 0.5 \
                     else (1 - proba) * 100

        results[model_name] = {
            'default_probability': round(float(proba * 100), 2),
            'decision':            decision,
           'confidence':          round(float(confidence), 2)        }

    return results


def print_results(borrower: dict, results: dict):
    """Print formatted prediction results"""
    print("\n" + "="*55)
    print("   CREDIT RISK ASSESSMENT REPORT")
    print("="*55)
    print("\nBORROWER PROFILE:")
    for key, val in borrower.items():
        print(f"  {key:<20} {val}")

    print("\n" + "-"*55)
    print("MODEL PREDICTIONS:")
    print("-"*55)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Default Probability: "
              f"{result['default_probability']}%")
        print(f"  Decision:            "
              f"{result['decision']}")
        print(f"  Confidence:          "
              f"{result['confidence']}%")

    print("\n" + "="*55)

    # Majority vote
    decisions = [r['decision'] for r in results.values()]
    high_risk_votes = sum(
        1 for d in decisions if 'HIGH' in d
    )

    print("FINAL RECOMMENDATION (Majority Vote):")
    if high_risk_votes >= 2:
        print("  ⚠️  HIGH RISK — LOAN APPLICATION DECLINED")
        print("  Reason: Majority of models predict default")
    else:
        print("  ✅  LOW RISK — LOAN APPLICATION APPROVED")
        print("  Reason: Majority of models predict repayment")
    print("="*55)


# ===============================
# TEST WITH SAMPLE BORROWERS
# ===============================
if __name__ == "__main__":

    print("=" * 55)
    print("   CREDIT RISK PREDICTION PIPELINE")
    print("   Author: Thabiso Mdaka | UKZN")
    print("=" * 55)

    # --- Borrower 1: High Risk ---
    borrower_1 = {
        'Age':              25,
        'Sex':              'male',
        'Job':              1,
        'Housing':          'free',
        'Saving accounts':  'little',
        'Checking account': 'little',
        'Credit amount':    15000,
        'Duration':         48,
        'Purpose':          'vacation/others'
    }

    print("\n TEST 1 — Expected: HIGH RISK")
    results_1 = predict_risk(borrower_1)
    print_results(borrower_1, results_1)

    # --- Borrower 2: Low Risk ---
    borrower_2 = {
        'Age':              52,
        'Sex':              'male',
        'Job':              3,
        'Housing':          'own',
        'Saving accounts':  'rich',
        'Checking account': 'rich',
        'Credit amount':    2000,
        'Duration':         12,
        'Purpose':          'car'
    }

    print("\n TEST 2 — Expected: LOW RISK")
    results_2 = predict_risk(borrower_2)
    print_results(borrower_2, results_2)

    # --- Borrower 3: Borderline ---
    borrower_3 = {
        'Age':              35,
        'Sex':              'female',
        'Job':              2,
        'Housing':          'own',
        'Saving accounts':  'moderate',
        'Checking account': 'little',
        'Credit amount':    5000,
        'Duration':         30,
        'Purpose':          'education'
    }

    print("\n TEST 3 — Expected: BORDERLINE")
    results_3 = predict_risk(borrower_3)
    print_results(borrower_3, results_3)