import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   CREDIT RISK — Final Evaluation Dashboard")
print("=" * 55)

# ===============================
# 1. LOAD & PREPARE
# ===============================
df = pd.read_csv("data/processed/cleaned_credit_data.csv")
df['Risk'] = ((df['Credit amount'] > 4000) &
              (df['Duration'] > 24)).astype(int)

X = df.drop(columns=['Risk'])
y = df['Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y
)

# Train models
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric='logloss', verbosity=0)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

lr_proba  = lr.predict_proba(X_test)[:, 1]
rf_proba  = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]

# ===============================
# 2. GINI COEFFICIENT
# ===============================
def gini(y_true, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    return (2 * auc - 1) * 100

lr_gini  = gini(y_test, lr_proba)
rf_gini  = gini(y_test, rf_proba)
xgb_gini = gini(y_test, xgb_proba)

# ===============================
# 3. KS STATISTIC
# ===============================
def ks_statistic(y_true, y_proba):
    good_scores = y_proba[y_true == 0]
    bad_scores  = y_proba[y_true == 1]
    ks = stats.ks_2samp(good_scores, bad_scores)
    return ks.statistic * 100

lr_ks  = ks_statistic(y_test, lr_proba)
rf_ks  = ks_statistic(y_test, rf_proba)
xgb_ks = ks_statistic(y_test, xgb_proba)

# ===============================
# 4. PRINT BANKING METRICS
# ===============================
print("\n" + "="*55)
print("   BANKING PERFORMANCE METRICS")
print("="*55)
print(f"\n{'Model':<25} {'ROC-AUC':>8} {'Gini':>8} {'KS':>8}")
print("-"*55)
print(f"{'Logistic Regression':<25} "
      f"{roc_auc_score(y_test,lr_proba):>8.4f} "
      f"{lr_gini:>7.1f}% "
      f"{lr_ks:>7.1f}%")
print(f"{'Random Forest':<25} "
      f"{roc_auc_score(y_test,rf_proba):>8.4f} "
      f"{rf_gini:>7.1f}% "
      f"{rf_ks:>7.1f}%")
print(f"{'XGBoost':<25} "
      f"{roc_auc_score(y_test,xgb_proba):>8.4f} "
      f"{xgb_gini:>7.1f}% "
      f"{xgb_ks:>7.1f}%")

print("\nBANKING BENCHMARKS:")
print("Gini > 40% = Good | Gini > 60% = Excellent")
print("KS   > 30% = Good | KS   > 40% = Excellent")

# ===============================
# 5. KS PLOT
# ===============================
print("\n--- Generating KS Plot ---")

def plot_ks(y_true, y_proba, model_name, ax):
    df_ks = pd.DataFrame({
        'score': y_proba,
        'target': y_true
    }).sort_values('score')

    total_good = (y_true == 0).sum()
    total_bad  = (y_true == 1).sum()

    df_ks['cum_good'] = (
        (df_ks['target'] == 0).cumsum() / total_good
    )
    df_ks['cum_bad'] = (
        (df_ks['target'] == 1).cumsum() / total_bad
    )
    df_ks['ks'] = abs(
        df_ks['cum_good'] - df_ks['cum_bad']
    )

    ks_max = df_ks['ks'].max() * 100
    ks_idx = df_ks['ks'].idxmax()
    ks_score = df_ks.loc[ks_idx, 'score']

    ax.plot(df_ks['score'], df_ks['cum_good'],
            color='#2196F3', linewidth=2,
            label='Cumulative Good (Low Risk)')
    ax.plot(df_ks['score'], df_ks['cum_bad'],
            color='#F44336', linewidth=2,
            label='Cumulative Bad (High Risk)')
    ax.axvline(x=ks_score, color='green',
               linestyle='--', linewidth=1.5,
               label=f'KS = {ks_max:.1f}%')
    ax.fill_between(df_ks['score'],
                    df_ks['cum_good'],
                    df_ks['cum_bad'],
                    alpha=0.1, color='green')
    ax.set_title(f'{model_name}\nKS Statistic = {ks_max:.1f}%',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Score Threshold')
    ax.set_ylabel('Cumulative Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_ks(y_test.values, lr_proba,  'Logistic Regression', axes[0])
plot_ks(y_test.values, rf_proba,  'Random Forest',       axes[1])
plot_ks(y_test.values, xgb_proba, 'XGBoost',             axes[2])

plt.suptitle('KS Statistic — Separation of Good vs Bad Borrowers\n'
             '(Banking Standard Risk Discrimination Metric)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/ks_plot.png', dpi=150)
plt.close()
print("Saved: outputs/plots/ks_plot.png")

# ===============================
# 6. FINAL SUMMARY DASHBOARD
# ===============================
print("--- Generating Summary Dashboard ---")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# ROC Curves
ax1 = fig.add_subplot(gs[0, :2])
for proba, label, color in zip(
        [lr_proba, rf_proba, xgb_proba],
        ['Logistic Regression', 'Random Forest', 'XGBoost'],
        ['#2196F3', '#4CAF50', '#F44336']):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax1.plot(fpr, tpr, linewidth=2.5,
             label=f'{label} (AUC={auc:.4f})',
             color=color)
ax1.plot([0,1],[0,1],'k--', linewidth=1)
ax1.set_title('ROC Curves — All Models',
              fontweight='bold', fontsize=12)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Model Comparison Bar
ax2 = fig.add_subplot(gs[0, 2])
models = ['LR', 'RF', 'XGB']
gini_vals = [lr_gini, rf_gini, xgb_gini]
ks_vals   = [lr_ks,   rf_ks,   xgb_ks]
x = np.arange(len(models))
width = 0.35
ax2.bar(x - width/2, gini_vals, width,
        label='Gini (%)', color='#2196F3',
        edgecolor='black')
ax2.bar(x + width/2, ks_vals, width,
        label='KS (%)', color='#F44336',
        edgecolor='black')
ax2.axhline(y=40, color='gray', linestyle='--',
            alpha=0.7, label='Good threshold (40%)')
ax2.set_title('Gini & KS Comparison',
              fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel('Score (%)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Score Distribution
ax3 = fig.add_subplot(gs[1, :2])
ax3.hist(lr_proba[y_test==0], bins=30,
         alpha=0.7, color='#2196F3',
         label='Low Risk', density=True)
ax3.hist(lr_proba[y_test==1], bins=30,
         alpha=0.7, color='#F44336',
         label='High Risk', density=True)
ax3.axvline(x=0.5, color='black',
            linestyle='--', linewidth=2,
            label='Decision Threshold (0.5)')
ax3.set_title('Score Distribution — Good vs Bad Borrowers',
              fontweight='bold', fontsize=12)
ax3.set_xlabel('Predicted Default Probability')
ax3.set_ylabel('Density')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# IV Summary Table
ax4 = fig.add_subplot(gs[1, 2])
iv_data = {
    'Duration':       9.64,
    'Credit Amt':     7.38,
    'Purpose':        0.31,
    'Housing':        0.14,
    'Check. Acct':    0.05,
    'Saving Acct':    0.05,
    'Sex':            0.03,
    'Age':            0.01
}
iv_series = pd.Series(iv_data).sort_values()
colors_iv = ['#F44336' if v >= 0.5
             else '#4CAF50' if v >= 0.3
             else '#2196F3' if v >= 0.1
             else '#FFC107' if v >= 0.02
             else '#9E9E9E'
             for v in iv_series.values]
ax4.barh(iv_series.index, iv_series.values,
         color=colors_iv, edgecolor='black',
         linewidth=0.5)
ax4.set_title('IV — Feature Power',
              fontweight='bold', fontsize=12)
ax4.set_xlabel('Information Value')
ax4.grid(True, alpha=0.3, axis='x')

plt.suptitle('Credit Risk Scorecard — Complete Evaluation Dashboard\n'
             'Thabiso Mdaka | BSc Electronic Engineering | UKZN',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/evaluation_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/evaluation_dashboard.png")

# ===============================
# 7. SAVE FINAL REPORT
# ===============================
report = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'ROC_AUC': [
        round(roc_auc_score(y_test, lr_proba), 4),
        round(roc_auc_score(y_test, rf_proba), 4),
        round(roc_auc_score(y_test, xgb_proba), 4)
    ],
    'Gini_%': [round(lr_gini,1),
               round(rf_gini,1),
               round(xgb_gini,1)],
    'KS_%':   [round(lr_ks,1),
               round(rf_ks,1),
               round(xgb_ks,1)]
})
report.to_csv('outputs/reports/final_evaluation.csv',
              index=False)

print("\n" + "="*55)
print("   COMPLETE — All outputs saved!")
print("="*55)
print("\nOutputs generated:")
print("  outputs/plots/ks_plot.png")
print("  outputs/plots/evaluation_dashboard.png")
print("  outputs/reports/final_evaluation.csv")
print("="*55)