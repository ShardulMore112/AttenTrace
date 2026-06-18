import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
PHASE2_RESULTS_PATH = "pilot_results.json"  
PHASE1_RESULTS_PATH = "phase1_results.json" 

def get_ground_truth_labels(phase1_path):
    """
    STEP 1: FETCH THE DATA
    Reads the Phase 1 JSON file and creates a dictionary 
    mapping the problem ID to its true/false correctness label.
    """
    with open(phase1_path, "r", encoding="utf-8") as f:
        phase1_data = json.load(f)
        
    labels = {}
    missing_correct_count = 0
    missing_id_count = 0
    
    for item in phase1_data:
        # If the entry is a summary block or corrupted and has no ID, skip it entirely.
        if "id" not in item:
            missing_id_count += 1
            continue
            
        # Robust fetch: If 'is_correct' is missing, assume False (0).
        if "is_correct" not in item:
            missing_correct_count += 1
            
        is_correct = item.get("is_correct", False)
        labels[item["id"]] = 1 if is_correct else 0 
        
    if missing_id_count > 0:
        print(f"[DEBUG] Ignored {missing_id_count} garbage entries in Phase 1 missing an 'id'.")
    if missing_correct_count > 0:
        print(f"[DEBUG] Found {missing_correct_count} problems missing the 'is_correct' key. Assumed Incorrect (0).")
        
    return labels

def load_and_prepare_data():
    """
    STEP 2: MERGE THE DATA
    Combines the labels from Phase 1 with the features from Phase 2.
    """
    with open(PHASE2_RESULTS_PATH, "r", encoding="utf-8") as f:
        phase2_data = json.load(f)
        
    labels = get_ground_truth_labels(PHASE1_RESULTS_PATH)
        
    df_rows = []
    for item in phase2_data:
        # If a problem exists in Kaggle but we don't have a label for it, skip it safely
        if item["id"] not in labels:
            continue
            
        target = labels[item["id"]]
        
        row = {
            "id": item["id"],
            "target": target,
            "total_seq_len": item["total_seq_len"],
            
            # --- FEATURE SET A: OUTPUT ONLY ---
            "out_tau": item["out_tau"],
            "out_mean": np.mean(item["step_out_mean"]),
            "out_max": np.max(item["step_out_mean"]),
            
            # --- FEATURE SET B: ATTENTION ONLY (Normalized) ---
            "attn_tau": item["attn_tau"],
            "attn_mean": np.mean(item["step_attn_global_norm"]),
            "attn_max": np.max(item["step_attn_global_norm"]),
            "attn_std_mean": np.mean(item["step_attn_std"]),
            
            # --- LAYER-SPECIFIC EFFECTS ---
            "layer_27_mean": np.mean(item["layer_wise_attn"]["layer_27"]),
        }
        df_rows.append(row)
        
    return pd.DataFrame(df_rows)

def evaluate_incremental_value():
    """
    STEP 3: EVALUATE (The Machine Learning Phase)
    """
    df = load_and_prepare_data()
    
    print(f"Successfully aligned and loaded {len(df)} complete trajectories.")
    if len(df) < 10:
        print("[WARNING] Dataset too small for reliable cross-validation. Found only:", len(df))
        return
        
    # Define the competing feature sets
    features_output = ["out_tau", "out_mean", "out_max", "total_seq_len"]
    features_attn = ["attn_tau", "attn_mean", "attn_max", "attn_std_mean", "layer_27_mean", "total_seq_len"]
    features_combined = list(set(features_output + features_attn))
    
    # --- THE FIX: Force all feature columns to float64 to prevent Pandas LossySetitemError ---
    df[features_combined] = df[features_combined].astype(float)
    
    X = df
    y = df["target"].values
    
    # Stratified K-Fold ensures equal class balance in testing
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_a, auc_b, auc_c = [], [], []
    scaler = StandardScaler()
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features to help Logistic Regression converge properly
        X_train.loc[:, features_combined] = scaler.fit_transform(X_train[features_combined])
        X_val.loc[:, features_combined] = scaler.transform(X_val[features_combined])
        
        # Model A: Output Only (Your Phase 1 Baseline)
        clf_a = LogisticRegression(max_iter=1000)
        clf_a.fit(X_train[features_output], y_train)
        auc_a.append(roc_auc_score(y_val, clf_a.predict_proba(X_val[features_output])[:, 1]))
        
        # Model B: Attention Only
        clf_b = LogisticRegression(max_iter=1000)
        clf_b.fit(X_train[features_attn], y_train)
        auc_b.append(roc_auc_score(y_val, clf_b.predict_proba(X_val[features_attn])[:, 1]))
        
        # Model C: Combined (Output + Attention)
        clf_c = LogisticRegression(max_iter=1000)
        clf_c.fit(X_train[features_combined], y_train)
        auc_c.append(roc_auc_score(y_val, clf_c.predict_proba(X_val[features_combined])[:, 1]))
        
    print(f"\n" + "="*55)
    print(f"       PHASE 3: INCREMENTAL PREDICTIVE VALUE       ")
    print(f"="*55)
    print(f"Model A (Output Only) ROC-AUC:       {np.mean(auc_a):.4f}  (±{np.std(auc_a):.3f})")
    print(f"Model B (Attention Only) ROC-AUC:    {np.mean(auc_b):.4f}  (±{np.std(auc_b):.3f})")
    print(f"Model C (Combined Features) ROC-AUC: {np.mean(auc_c):.4f}  (±{np.std(auc_c):.3f})")
    print(f"-"*55)
    
    delta = np.mean(auc_c) - np.mean(auc_a)
    print(f"NET SIGNAL GAIN (C vs A): {delta:+.4f} ROC-AUC points.")
    print(f"="*55)

if __name__ == "__main__":
    evaluate_incremental_value()