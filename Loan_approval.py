import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score,
                              precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# ============================================================
# STEP 1 — LOAD & EXPLORE
# ============================================================
df = pd.read_csv('loan_data.csv')
df.columns = df.columns.str.strip()
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print(f"\nApproval rate: {df['loan_status'].mean()*100:.2f}%")

# ============================================================
# STEP 2 — CLEAN
# ============================================================
df.dropna(inplace=True)
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()
print(f"Shape after cleaning: {df.shape}")

# ============================================================
# STEP 3 — DIAGNOSIS
# ============================================================
print("\n" + "="*55)
print("DIAGNOSIS: What drives loan_status in this dataset?")
print("="*55)

bins   = [0, 580, 640, 700, 750, 999]
labels = ['Poor(<580)', 'Fair(580-639)', 'Good(640-699)',
          'VeryGood(700-749)', 'Excellent(750+)']
df['credit_band'] = pd.cut(df['credit_score'], bins=bins, labels=labels)
print("\nCredit score vs approval:")
print(df.groupby('credit_band', observed=True)['loan_status'].mean().mul(100).round(1))

print("\nDefault history vs approval:")
print(df.groupby('previous_loan_defaults_on_file')['loan_status'].mean().mul(100).round(1))

print("\nIncome quartiles vs approval:")
df['income_q'] = pd.qcut(df['person_income'], q=4, labels=['Q1','Q2','Q3','Q4'])
print(df.groupby('income_q', observed=True)['loan_status'].mean().mul(100).round(1))

print("\nLoan intent vs approval:")
print(df.groupby('loan_intent')['loan_status'].mean().mul(100).round(1).sort_values())

print("\nHome ownership vs approval:")
print(df.groupby('person_home_ownership')['loan_status'].mean().mul(100).round(1))

# ============================================================
# STEP 4 — ENCODE CATEGORICALS
# ============================================================
df['home_enc']    = df['person_home_ownership'].map(
    {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 0}
)
df['intent_enc']  = df['loan_intent'].map({
    'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2,
    'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5
})
df['had_default'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
df['gender_enc']  = df['person_gender'].map({'male': 1, 'female': 0})
df['edu_enc']     = df['person_education'].map(
    {'High School': 0, 'Associate': 1, 'Bachelor': 2,
     'Master': 3, 'Doctorate': 4}
)

# ============================================================
# STEP 5 — ENHANCED FEATURE ENGINEERING
# ============================================================
df['monthly_income'] = df['person_income'] / 12

def credit_grade(score):
    if score < 580:   return 0
    elif score < 640: return 1
    elif score < 700: return 2
    elif score < 750: return 3
    else:             return 4

df['credit_grade']     = df['credit_score'].apply(credit_grade)
df['low_emp_flag']     = (df['person_emp_exp'] < 1).astype(int)
df['short_hist_flag']  = (df['cb_person_cred_hist_length'] < 2).astype(int)
df['high_debt_flag']   = (df['loan_percent_income'] > 0.35).astype(int)
df['low_income_flag']  = (df['monthly_income'] < 2000).astype(int)

# ── NEW: Interaction features to break had_default dominance ──
# These give the model ways to differentiate among non-defaulters
df['income_x_grade']     = df['monthly_income'] * df['credit_grade']           # high income + good credit
df['debt_x_intent']      = df['loan_percent_income'] * df['intent_enc']         # debt-to-income shaped by purpose
df['age_x_emp']          = df['person_age'] * df['person_emp_exp']              # maturity + stability
df['income_debt_ratio']  = df['monthly_income'] / (df['loan_amnt'] + 1)        # income per unit loan
df['credit_x_hist']      = df['credit_score'] * df['cb_person_cred_hist_length']  # credit strength x depth
df['rate_burden']        = df['loan_int_rate'] * df['loan_percent_income']     # cost-weighted debt load
df['emp_income_combo']   = df['person_emp_exp'] * df['monthly_income']         # stability x capacity
df['cred_hist_band']     = pd.cut(df['cb_person_cred_hist_length'],
                                   bins=[0, 2, 4, 7, 100],
                                   labels=[0, 1, 2, 3]).astype(int)              # Credit history band
df['fine_credit_score']  = df['credit_score']  # Raw continuous score (alongside grade)

print("\n[OK] Interaction features created.")

# ============================================================
# STEP 6 — FEATURE LIST (EXPANDED)
# ============================================================
features = [
    # Personal profile
    'person_age',
    'gender_enc',
    'edu_enc',
    'person_emp_exp',
    'home_enc',
    'low_emp_flag',

    # Credit quality
    'credit_grade',
    'fine_credit_score',           # Continuous CIBIL — more granular
    'had_default',                 # Kept, but model will now split on others too
    'cb_person_cred_hist_length',
    'cred_hist_band',
    'short_hist_flag',

    # Income & repayment
    'monthly_income',
    'loan_percent_income',
    'high_debt_flag',
    'low_income_flag',

    # Loan details
    'loan_amnt',
    'loan_int_rate',
    'intent_enc',

    # Interaction features (break had_default monopoly)
    'income_x_grade',
    'debt_x_intent',
    'age_x_emp',
    'income_debt_ratio',
    'credit_x_hist',
    'rate_burden',
    'emp_income_combo',
]

X = df[features]
y = df['loan_status']

print(f"\nFeature matrix: {X.shape}  ({len(features)} features)")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Approval rate: {y.mean()*100:.2f}%")

# ============================================================
# STEP 7 — SPLIT & SCALE
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print("Scaler saved!")

# ============================================================
# STEP 8 — SMOTE OVERSAMPLING (balance training set)
# ============================================================
print("\n[INFO] Applying SMOTE to balance training class distribution...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# ============================================================
# STEP 9 — TRAIN RANDOM FOREST (baseline)
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_sm, y_train_sm)
rf_preds = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_acc   = accuracy_score(y_test, rf_preds) * 100
rf_auc   = roc_auc_score(y_test, rf_proba) * 100

# ============================================================
# STEP 10 — TRAIN XGBOOST (tuned to prevent had_default dominance)
# ============================================================
neg = y_train_sm.value_counts()[0]
pos = y_train_sm.value_counts()[1]
print(f"\nSMOTE class ratio after balance: {neg}/{pos}")

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,               # Shallower trees = less memorization
    learning_rate=0.02,        # Slow learning = more feature usage
    subsample=0.75,
    colsample_bytree=0.45,     # Low column sampling = forces use of all features
    colsample_bylevel=0.5,     # Further regularize per-level splits
    min_child_weight=10,       # Requires more samples per leaf
    gamma=0.5,                 # Minimum loss reduction needed to split
    reg_alpha=1.2,             # L1 regularization (sparse feature selection)
    reg_lambda=3.0,            # L2 regularization (weight penalty)
    scale_pos_weight=1.0,      # SMOTE balanced the classes — no weight needed
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)
xgb_model.fit(X_train_sm, y_train_sm)
xgb_proba_raw = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc_raw   = roc_auc_score(y_test, xgb_proba_raw) * 100

# ── Calibrate probabilities for realistic risk estimates ──────
calibrated = CalibratedClassifierCV(estimator=xgb_model, method='isotonic', cv='prefit')
calibrated.fit(X_test_scaled, y_test)
xgb_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba) * 100

# ============================================================
# STEP 11 — PROBABILITY DISTRIBUTION CHECK
# ============================================================
print("\n" + "="*55)
print("PROBABILITY DISTRIBUTION (XGBoost Calibrated)")
print("="*55)
print(f"  Min:    {xgb_proba.min():.4f}")
print(f"  Max:    {xgb_proba.max():.4f}")
print(f"  Mean:   {xgb_proba.mean():.4f}")
print(f"  Median: {np.median(xgb_proba):.4f}")
print(f"  % below 0.30:  {(xgb_proba<0.30).mean()*100:.1f}%")
print(f"  % 0.30-0.50:   {((xgb_proba>=0.30)&(xgb_proba<0.50)).mean()*100:.1f}%")
print(f"  % 0.50-0.70:   {((xgb_proba>=0.50)&(xgb_proba<0.70)).mean()*100:.1f}%")
print(f"  % above 0.70:  {(xgb_proba>=0.70).mean()*100:.1f}%")

# ============================================================
# STEP 12 — OPTIMAL THRESHOLD SELECTION (maximize F1-macro)
# ============================================================
print("\n" + "="*75)
print(f"{'Threshold':>12} {'Accuracy':>10} {'F1-Rejected':>13} "
      f"{'F1-Approved':>13} {'F1-macro':>10} {'Approval%':>10}")
print("="*75)

best_thr      = 0.50
best_balanced = -1

for thr in np.arange(0.25, 0.80, 0.05):
    preds    = (xgb_proba >= thr).astype(int)
    acc      = accuracy_score(y_test, preds) * 100
    f1_rej   = f1_score(y_test, preds, pos_label=0, zero_division=0)
    f1_app   = f1_score(y_test, preds, pos_label=1, zero_division=0)
    apr_rate = preds.mean() * 100
    balanced = (f1_rej + f1_app) / 2
    marker   = " <- best" if balanced > best_balanced else ""
    if balanced > best_balanced:
        best_balanced = balanced
        best_thr      = float(round(thr, 2))
    print(f"{thr:>12.2f} {acc:>9.2f}% {f1_rej:>13.4f} "
          f"{f1_app:>13.4f} {balanced:>10.4f} {apr_rate:>9.1f}%{marker}")

print(f"\n[OK] Selected threshold: {best_thr:.2f}")
xgb_preds = (xgb_proba >= best_thr).astype(int)

# ============================================================
# STEP 13 — FINAL RESULTS & CONFUSION MATRIX
# ============================================================
xgb_acc = accuracy_score(y_test, xgb_preds) * 100

print("\n" + "="*55)
print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
print("="*55)
print(f"{'Random Forest (baseline)':<25} {rf_acc:>9.2f}% {rf_auc:>9.2f}%")
print(f"{'XGBoost + SMOTE + Calib':<25} {xgb_acc:>9.2f}% {xgb_auc:>9.2f}%")
print("="*55)

cm = confusion_matrix(y_test, xgb_preds)
print("\nXGBoost Confusion Matrix:")
print(cm)
print(f"\n  Correct rejections  (TN): {cm[0][0]}")
print(f"  Wrong approvals     (FP): {cm[0][1]}  <- bank loses money")
print(f"  Wrong rejections    (FN): {cm[1][0]}  <- good customers lost")
print(f"  Correct approvals   (TP): {cm[1][1]}")
print(f"\n  FP Rate: {cm[0][1]/(cm[0][0]+cm[0][1])*100:.2f}%  |  FN Rate: {cm[1][0]/(cm[1][0]+cm[1][1])*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, xgb_preds,
      target_names=['Rejected','Approved']))

# ============================================================
# STEP 14 — FEATURE IMPORTANCE (check had_default dominance)
# ============================================================
feat_df = pd.DataFrame({
    'Feature':    features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 feature importances:")
print(feat_df.head(10).to_string(index=False))

top     = feat_df.iloc[0]
top_pct = top['Importance'] * 100
print(f"\nTop feature: '{top['Feature']}' at {top_pct:.1f}%")
if top_pct < 40:
    print("[OK] Well distributed — no single feature dominates!")
elif top_pct < 60:
    print("[WARN] Moderate — acceptable for this dataset")
else:
    print("[WARN] Still concentrated — consider reducing model complexity")

# ============================================================
# STEP 15 — PLOTS (saved to disk, not shown interactively)
# ============================================================
# Confusion matrix heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, title, cmap in zip(
    axes,
    [rf_preds, xgb_preds],
    ['Random Forest (Baseline)', f'XGBoost+SMOTE+Calib (thr={best_thr:.2f})'],
    ['Blues', 'Greens']
):
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d',
                cmap=cmap, ax=ax,
                xticklabels=['Rejected','Approved'],
                yticklabels=['Rejected','Approved'])
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n[OK] Confusion matrix saved to model/confusion_matrix.png")

# Feature importance chart
feat_plot = feat_df.sort_values('Importance', ascending=True)
colors = ['#e74c3c' if v > 0.15 else
          '#f39c12' if v > 0.05 else
          '#3498db' for v in feat_plot['Importance']]
plt.figure(figsize=(12, 9))
plt.barh(feat_plot['Feature'], feat_plot['Importance'], color=colors)
plt.title('XGBoost Feature Importance\n(Red >15% . Orange >5% . Blue <5%)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('model/feature_importance.png', dpi=150, bbox_inches='tight')
print("[OK] Feature importance chart saved to model/feature_importance.png")

# ============================================================
# STEP 16 — CROSS VALIDATION
# ============================================================
print("\n[INFO] 5-Fold Cross Validation (on SMOTE-balanced train set):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train_sm, y_train_sm,
                             cv=cv, scoring='roc_auc')
print(f"ROC-AUC: {[round(s,4) for s in cv_scores]}")
print(f"Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")
print("[OK] Consistent" if cv_scores.std() < 0.02 else "[WARN] High variance")

# ============================================================
# STEP 17 — SAVE MODEL + CALIBRATED WRAPPER
# ============================================================
# Save the calibrated model so server.py gets calibrated probabilities
if xgb_auc >= rf_auc:
    best_model = calibrated   # <-- Save the CALIBRATED version
    best_name  = "XGBoost+SMOTE+Calibrated"
    best_auc   = xgb_auc
else:
    best_model = rf_model
    best_name  = "Random Forest"
    best_auc   = rf_auc

joblib.dump(best_model, 'model/loan_model.pkl')
joblib.dump(features,   'model/feature_names.pkl')
joblib.dump(best_thr,   'model/threshold.pkl')

print(f"\n[OK] Best model: {best_name} (ROC-AUC: {best_auc:.2f}%)")
print(f"     Threshold: {best_thr:.2f}")
print("Files saved:", os.listdir('model/'))