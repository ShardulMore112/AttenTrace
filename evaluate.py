import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

FILE_PATH = "phase2_math500_results.json"

# 1. Load Data
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Extract Features
records = []
for d in data:
    if len(d["step_attn_std"]) > 0:
        mean_attn_std = np.mean(d["step_attn_std"])
    else:
        mean_attn_std = 0.0
        
    records.append({
        "id": d["id"],
        "is_correct": d["is_correct"],
        "out_tau": d.get("out_tau", 0.0),
        "mean_attn_std": mean_attn_std
    })

df = pd.DataFrame(records)
df = df.dropna()

# 3. Print Base Yield and Accuracy
print(f"--- MATH-500 DATASET STATS ---")
print(f"Valid Trajectories: {len(df)}")
print(f"Base Accuracy: {(df['is_correct'].mean() * 100):.1f}%\n")

# 4. Repeated Stratified CV (5 Splits, 10 Repeats = 50 Folds)
X_base = df[['out_tau']]
X_prop = df[['out_tau', 'mean_attn_std']]
y = df['is_correct']

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

base_aucs = []
prop_aucs = []
deltas = []
wins = 0

for train_idx, test_idx in rskf.split(X_base, y):
    # Base Model
    clf_base = LogisticRegression(class_weight='balanced')
    clf_base.fit(X_base.iloc[train_idx], y.iloc[train_idx])
    preds_base = clf_base.predict_proba(X_base.iloc[test_idx])[:, 1]
    auc_base = roc_auc_score(y.iloc[test_idx], preds_base)
    
    # Proposed Model
    clf_prop = LogisticRegression(class_weight='balanced')
    clf_prop.fit(X_prop.iloc[train_idx], y.iloc[train_idx])
    preds_prop = clf_prop.predict_proba(X_prop.iloc[test_idx])[:, 1]
    auc_prop = roc_auc_score(y.iloc[test_idx], preds_prop)
    
    base_aucs.append(auc_base)
    prop_aucs.append(auc_prop)
    delta = auc_prop - auc_base
    deltas.append(delta)
    
    if delta > 0:
        wins += 1

# 5. Calculate Confidence Intervals
deltas = np.array(deltas)
mean_delta = np.mean(deltas)
ci_lower = np.percentile(deltas, 2.5)
ci_upper = np.percentile(deltas, 97.5)
win_rate = (wins / len(deltas)) * 100

print(f"--- REPEATED CV RESULTS (50 Folds) ---")
print(f"Mean Base AUC:     {np.mean(base_aucs):.4f}")
print(f"Mean Proposed AUC: {np.mean(prop_aucs):.4f}")
print(f"Mean AUC Delta:   +{mean_delta:.4f}")
print(f"95% CI of Delta:  [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Fold Win Rate:     {win_rate:.1f}%")

from scipy.stats import ttest_1samp
t_stat, p_value = ttest_1samp(deltas, 0.0, alternative='greater')
print(f"Paired T-Test p-value (One-Sided): {p_value:.2e}")

print(f"Raw deltas array: {list(deltas)}")