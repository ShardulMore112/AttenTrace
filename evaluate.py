import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon

# --- CONFIGURATION ---
# Using the exact feature names extracted from your Kaggle pipeline
OUTPUT_FEATURES = ['out_tau', 'step_out_mean']
ATTN_FEATURES = ['attn_tau', 'step_attn_global', 'step_attn_global_norm', 'step_attn_std']
COMBINED_FEATURES = OUTPUT_FEATURES + ATTN_FEATURES

DATASETS = [
    {"name": "GSM8K", "path": "phase2_gsm8k_1000_cleaned.json"},
    {"name": "MATH-500", "path": "phase2_math500_results.json"}
]

def load_and_clean_data(filepath):
    """Loads JSON data, averages lists into scalars, and sets up the target."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        df = df.dropna(subset=COMBINED_FEATURES + ['is_correct'])
        
        # Predict hallucination/failure (1) vs success (0)
        df['is_failure'] = 1 - df['is_correct']
        
        # Convert any attention arrays into scalar means
        for col in COMBINED_FEATURES:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else x)
                
        df = df.dropna(subset=COMBINED_FEATURES)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def evaluate_dataset(df, dataset_name):
    print(f"\n========================================")
    print(f"--- {dataset_name} DATASET STATS ---")
    print(f"Valid Trajectories: {len(df)}")
    print(f"Base Accuracy: {(df['is_correct'].mean() * 100):.1f}%")
    print(f"Base Failure Rate: {(df['is_failure'].mean() * 100):.1f}%\n")

    X_base = df[OUTPUT_FEATURES]
    X_prop = df[COMBINED_FEATURES]
    y = df['is_failure']

    # 5 Splits, 10 Repeats = 50 Folds
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    base_aucs = []
    prop_aucs = []
    deltas = []
    wins = 0

    # Pipeline prevents data leakage by scaling inside the CV folds
    pipeline_base = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    pipeline_prop = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    for train_idx, test_idx in rskf.split(X_base, y):
        # Base Model (Output Features Only)
        pipeline_base.fit(X_base.iloc[train_idx], y.iloc[train_idx])
        preds_base = pipeline_base.predict_proba(X_base.iloc[test_idx])[:, 1]
        auc_base = roc_auc_score(y.iloc[test_idx], preds_base)
        
        # Proposed Model (Combined Features)
        pipeline_prop.fit(X_prop.iloc[train_idx], y.iloc[train_idx])
        preds_prop = pipeline_prop.predict_proba(X_prop.iloc[test_idx])[:, 1]
        auc_prop = roc_auc_score(y.iloc[test_idx], preds_prop)
        
        base_aucs.append(auc_base)
        prop_aucs.append(auc_prop)
        
        delta = auc_prop - auc_base
        deltas.append(delta)
        
        if delta > 0:
            wins += 1

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

    # Wilcoxon signed-rank test (Reviewer 3's requested non-parametric test)
    stat, p_value = wilcoxon(deltas, alternative='greater')
    print(f"Wilcoxon p-value (One-Sided): {p_value:.2e}")
    print(f"========================================")

if __name__ == "__main__":
    for ds in DATASETS:
        df = load_and_clean_data(ds["path"])
        if df is not None:
            evaluate_dataset(df, ds["name"])