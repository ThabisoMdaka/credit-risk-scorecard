import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===============================
# WEIGHT OF EVIDENCE & INFORMATION VALUE
# Author: Thabiso Mdaka
# ===============================
print("=" * 55)
print("   CREDIT RISK — WoE & IV Analysis")
print("=" * 55)

# ===============================
# 1. LOAD DATA
# ===============================
df_raw = pd.read_csv("data/raw/german_credit.csv")
df_raw.drop(columns=['Unnamed: 0'], inplace=True)

# Fill missing values
df_raw['Saving accounts'] = df_raw['Saving accounts'].fillna('unknown')
df_raw['Checking account'] = df_raw['Checking account'].fillna('unknown')

# Create target variable
df_raw['Risk'] = ((df_raw['Credit amount'] > 4000) &
                  (df_raw['Duration'] > 24)).astype(int)

print(f"\nDataset: {len(df_raw)} borrowers")
print(f"High Risk: {df_raw['Risk'].sum()} ({df_raw['Risk'].mean()*100:.1f}%)")
print(f"Low Risk:  {(df_raw['Risk']==0).sum()} ({(1-df_raw['Risk'].mean())*100:.1f}%)")

# ===============================
# 2. WoE & IV CALCULATION FUNCTION
# ===============================
def calculate_woe_iv(df, feature, target):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV)
    
    WoE = ln(%Good / %Bad)
    IV  = Sum((%Good - %Bad) * WoE)
    
    WoE > 0: Category has more Good borrowers
    WoE < 0: Category has more Bad borrowers
    
    IV Interpretation:
    < 0.02  : Not predictive
    0.02-0.1: Weak predictor
    0.1-0.3 : Medium predictor
    0.3-0.5 : Strong predictor
    > 0.5   : Very strong predictor
    """
    total_good = (df[target] == 0).sum()
    total_bad  = (df[target] == 1).sum()

    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad']
    grouped['Good'] = grouped['Total'] - grouped['Bad']

    grouped['pct_Good'] = grouped['Good'] / total_good
    grouped['pct_Bad']  = grouped['Bad']  / total_bad

    # Avoid division by zero
    grouped['pct_Good'] = grouped['pct_Good'].replace(0, 0.0001)
    grouped['pct_Bad']  = grouped['pct_Bad'].replace(0, 0.0001)

    grouped['WoE'] = np.log(grouped['pct_Good'] / grouped['pct_Bad'])
    grouped['IV']  = (grouped['pct_Good'] - grouped['pct_Bad']) * grouped['WoE']

    iv_total = grouped['IV'].sum()

    return grouped.reset_index(), iv_total


# ===============================
# 3. CALCULATE WoE/IV FOR CATEGORICAL FEATURES
# ===============================
categorical_features = [
    'Sex', 'Housing', 'Saving accounts',
    'Checking account', 'Purpose'
]

iv_summary = {}

print("\n--- Calculating WoE & IV ---\n")
for feature in categorical_features:
    woe_df, iv = calculate_woe_iv(df_raw, feature, 'Risk')
    iv_summary[feature] = iv
    print(f"{feature:<20} IV = {iv:.4f}  ", end="")
    if iv < 0.02:
        print("→ Not Predictive")
    elif iv < 0.1:
        print("→ Weak Predictor")
    elif iv < 0.3:
        print("→ Medium Predictor")
    elif iv < 0.5:
        print("→ Strong Predictor")
    else:
        print("→ Very Strong Predictor ")

# ===============================
# 4. NUMERICAL FEATURES — BINNED WoE
# ===============================
numerical_features = ['Age', 'Credit amount', 'Duration']

print("\n--- Binning Numerical Features ---")
for feature in numerical_features:
    df_raw[f'{feature}_bin'] = pd.qcut(
        df_raw[feature], q=5,
        duplicates='drop',
        labels=False
    )
    woe_df, iv = calculate_woe_iv(
        df_raw, f'{feature}_bin', 'Risk'
    )
    iv_summary[feature] = iv
    print(f"{feature:<20} IV = {iv:.4f}  ", end="")
    if iv < 0.02:
        print("→ Not Predictive")
    elif iv < 0.1:
        print("→ Weak Predictor")
    elif iv < 0.3:
        print("→ Medium Predictor")
    elif iv < 0.5:
        print("→ Strong Predictor")
    else:
        print("→ Very Strong Predictor ")

# ===============================
# 5. VISUALIZATIONS
# ===============================
print("\n--- Generating WoE/IV Plots ---")

# Plot 1: Information Value Ranking
iv_df = pd.DataFrame(
    list(iv_summary.items()),
    columns=['Feature', 'IV']
).sort_values('IV', ascending=True)

colors = []
for iv in iv_df['IV']:
    if iv < 0.02:
        colors.append('#9E9E9E')
    elif iv < 0.1:
        colors.append('#FFC107')
    elif iv < 0.3:
        colors.append('#2196F3')
    elif iv < 0.5:
        colors.append('#4CAF50')
    else:
        colors.append('#F44336')

plt.figure(figsize=(12, 8))
bars = plt.barh(iv_df['Feature'], iv_df['IV'],
                color=colors, edgecolor='black',
                linewidth=0.5)

# Add value labels
for bar, iv in zip(bars, iv_df['IV']):
    plt.text(bar.get_width() + 0.01,
             bar.get_y() + bar.get_height()/2,
             f'{iv:.4f}', va='center',
             fontweight='bold', fontsize=10)

# Add threshold lines
plt.axvline(x=0.02, color='gray', linestyle='--',
            alpha=0.7, label='Weak (0.02)')
plt.axvline(x=0.1,  color='#FFC107', linestyle='--',
            alpha=0.7, label='Medium (0.1)')
plt.axvline(x=0.3,  color='#2196F3', linestyle='--',
            alpha=0.7, label='Strong (0.3)')
plt.axvline(x=0.5,  color='#F44336', linestyle='--',
            alpha=0.7, label='Very Strong (0.5)')

plt.title('Information Value (IV) — Feature Predictive Power\n'
          'Banking Standard for Variable Selection',
          fontsize=13, fontweight='bold')
plt.xlabel('Information Value (IV)', fontsize=12)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/plots/information_value.png', dpi=150)
plt.close()
print("Saved: outputs/plots/information_value.png")

# Plot 2: WoE plots for top categorical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    woe_df, iv = calculate_woe_iv(df_raw, feature, 'Risk')
    colors_woe = ['#F44336' if w < 0 else '#2196F3'
                  for w in woe_df['WoE']]
    axes[idx].bar(woe_df[feature].astype(str),
                  woe_df['WoE'],
                  color=colors_woe,
                  edgecolor='black',
                  linewidth=0.5)
    axes[idx].axhline(y=0, color='black', linewidth=1.5)
    axes[idx].set_title(f'{feature}\n(IV = {iv:.4f})',
                        fontweight='bold', fontsize=11)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Weight of Evidence', fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(True, alpha=0.3, axis='y')

# Hide last empty subplot
axes[-1].set_visible(False)

plt.suptitle('Weight of Evidence (WoE) Analysis\n'
             'Red = Higher Risk | Blue = Lower Risk',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/woe_analysis.png', dpi=150)
plt.close()
print("Saved: outputs/plots/woe_analysis.png")

# Plot 3: WoE for numerical features (binned)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, feature in enumerate(numerical_features):
    woe_df, iv = calculate_woe_iv(
        df_raw, f'{feature}_bin', 'Risk'
    )
    colors_woe = ['#F44336' if w < 0 else '#2196F3'
                  for w in woe_df['WoE']]
    axes[idx].bar(
        woe_df[f'{feature}_bin'].astype(str),
        woe_df['WoE'],
        color=colors_woe,
        edgecolor='black',
        linewidth=0.5
    )
    axes[idx].axhline(y=0, color='black', linewidth=1.5)
    axes[idx].set_title(
        f'{feature} (Binned)\nIV = {iv:.4f}',
        fontweight='bold', fontsize=11
    )
    axes[idx].set_xlabel('Bin (0=Lowest, 4=Highest)',
                         fontsize=10)
    axes[idx].set_ylabel('Weight of Evidence', fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('WoE Analysis — Numerical Features (Quantile Binned)\n'
             'Red = Higher Risk | Blue = Lower Risk',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/woe_numerical.png', dpi=150)
plt.close()
print("Saved: outputs/plots/woe_numerical.png")

# ===============================
# 6. SAVE IV SUMMARY REPORT
# ===============================
iv_report = pd.DataFrame(
    list(iv_summary.items()),
    columns=['Feature', 'IV']
).sort_values('IV', ascending=False)

iv_report['Predictive Power'] = iv_report['IV'].apply(
    lambda x: 'Very Strong' if x >= 0.5
    else 'Strong' if x >= 0.3
    else 'Medium' if x >= 0.1
    else 'Weak' if x >= 0.02
    else 'Not Predictive'
)

iv_report.to_csv('outputs/reports/iv_summary.csv', index=False)

print("\n" + "="*55)
print("   IV SUMMARY REPORT")
print("="*55)
print(iv_report.to_string(index=False))
print("\nSaved: outputs/reports/iv_summary.csv")
print("="*55)