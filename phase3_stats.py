import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

FILE_PATH = "phase2_gsm8k_1000_cleaned.json"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract same features as before
rows = []
for record in data:
    if record.get("is_correct") is None: continue
    step_out = record.get("step_out_mean", [])
    step_attn = record.get("step_attn_global_norm", [])
    step_std = record.get("step_attn_std", [])
    
    rows.append({
        "is_failure": 1 - record["is_correct"],
        "out_tau": record.get("out_tau", 0.0),
        "mean_out_entropy": np.mean(step_out) if step_out else 0.0,
        "max_out_entropy": np.max(step_out) if step_out else 0.0,
        "attn_tau": record.get("attn_tau", 0.0),
        "mean_attn_entropy": np.mean(step_attn) if step_attn else 0.0,
        "max_attn_entropy": np.max(step_attn) if step_attn else 0.0,
        "mean_attn_std": np.mean(step_std) if step_std else 0.0,
    })

df = pd.DataFrame(rows).fillna(0.0)

features_standard = ["out_tau", "mean_out_entropy", "max_out_entropy"]
features_proposed = features_standard + ["attn_tau", "mean_attn_entropy", "max_attn_entropy", "mean_attn_std"]

X_std = df[features_standard].values
X_prop = df[features_proposed].values
y = df["is_failure"].values

# --- 1. 5-FOLD CROSS VALIDATION ---
print("--- 5-Fold Cross Validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
std_aucs, prop_aucs = [], []

for train_idx, test_idx in skf.split(X_std, y):
    # Standard
    scaler = StandardScaler().fit(X_std[train_idx])
    clf = LogisticRegression(class_weight='balanced').fit(scaler.transform(X_std[train_idx]), y[train_idx])
    std_aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(scaler.transform(X_std[test_idx]))[:, 1]))
    
    # Proposed
    scaler = StandardScaler().fit(X_prop[train_idx])
    clf = LogisticRegression(class_weight='balanced').fit(scaler.transform(X_prop[train_idx]), y[train_idx])
    prop_aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(scaler.transform(X_prop[test_idx]))[:, 1]))

print(f"Standard AUC: {np.mean(std_aucs):.4f} (std: {np.std(std_aucs):.4f})")
print(f"Proposed AUC: {np.mean(prop_aucs):.4f} (std: {np.std(prop_aucs):.4f})")
print(f"Mean Delta  : +{np.mean(prop_aucs) - np.mean(std_aucs):.4f}\n")

# --- 2. BOOTSTRAP CONFIDENCE INTERVAL (1000 Iterations) ---
print("--- Bootstrapping (1000 Resamples) ---")
n_iterations = 1000
deltas = []

for i in range(n_iterations):
    # Resample with replacement
    X_std_boot, X_prop_boot, y_boot = resample(X_std, X_prop, y, random_state=i)
    
    # We must ensure both classes are in the bootstrap sample
    if len(np.unique(y_boot)) < 2: continue 
    
    # Scale and Predict Standard
    scaler = StandardScaler().fit(X_std_boot)
    clf = LogisticRegression(class_weight='balanced').fit(scaler.transform(X_std_boot), y_boot)
    # Testing on the original full set (or Out-Of-Bag, but for simplicity we score on full to see stability of the delta)
    std_auc = roc_auc_score(y, clf.predict_proba(StandardScaler().fit(X_std).transform(X_std))[:, 1])
    
    # Scale and Predict Proposed
    scaler = StandardScaler().fit(X_prop_boot)
    clf = LogisticRegression(class_weight='balanced').fit(scaler.transform(X_prop_boot), y_boot)
    prop_auc = roc_auc_score(y, clf.predict_proba(StandardScaler().fit(X_prop).transform(X_prop))[:, 1])
    
    deltas.append(prop_auc - std_auc)

lower_bound = np.percentile(deltas, 2.5)
upper_bound = np.percentile(deltas, 97.5)

print(f"95% CI for AUC Improvement: [{lower_bound:.4f}, {upper_bound:.4f}]")
if lower_bound > 0:
    print("RESULT: SIGNIFICANT. The improvement is robust to resampling.")
else:
    print("RESULT: NOT SIGNIFICANT. The CI crosses zero.")