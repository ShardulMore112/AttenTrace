import os
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
# IMPORTANT: This must match the feature engineering used in your very first
# phase3_eval.py exactly, or the three rows of the master table are not comparable.
# mean_out_entropy / max_out_entropy are derived from step_out_mean (list -> mean/max),
# NOT the same single column as step_out_mean averaged alone.
RAW_LIST_FIELDS = {
    "step_out_mean": ["mean_out_entropy", "max_out_entropy"],          # -> mean, max
    "step_attn_global_norm": ["mean_attn_entropy", "max_attn_entropy"],  # -> mean, max
    "step_attn_std": ["mean_attn_std"],                                  # -> mean only
}

OUTPUT_FEATURES = ["out_tau", "mean_out_entropy", "max_out_entropy"]
ATTN_FEATURES = ["attn_tau", "mean_attn_entropy", "max_attn_entropy", "mean_attn_std"]
COMBINED_FEATURES = OUTPUT_FEATURES + ATTN_FEATURES

DATASETS = [
    {"name": "GSM8K (1.5B)", "path": "data\gs8mk\phase2_gsm8k_1000_cleaned.json"},
    {"name": "MATH-500 (1.5B)", "path": "data\math500\phase2_math500_results.json"},
    {"name": "MATH-500 (3B)", "path": "data\math500\phase2_math500_3B_graded.json"},
]


def load_and_clean_data(filepath):
    """Loads JSON, derives mean_out_entropy/max_out_entropy/mean_attn_entropy/
    max_attn_entropy/mean_attn_std from the raw per-step lists -- using the SAME
    derivation as the original phase3_eval.py -- so all three datasets are scored
    with an identical, comparable feature set."""
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}. Skipping...")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for record in data:
            if record.get("is_correct") is None:
                continue

            row = {"id": record.get("id"), "is_correct": record["is_correct"]}
            row["out_tau"] = record.get("out_tau", 0.0)
            row["attn_tau"] = record.get("attn_tau", 0.0)

            step_out = record.get("step_out_mean", [])
            step_attn_norm = record.get("step_attn_global_norm", [])
            step_std = record.get("step_attn_std", [])

            row["mean_out_entropy"] = float(np.mean(step_out)) if step_out else np.nan
            row["max_out_entropy"] = float(np.max(step_out)) if step_out else np.nan
            row["mean_attn_entropy"] = float(np.mean(step_attn_norm)) if step_attn_norm else np.nan
            row["max_attn_entropy"] = float(np.max(step_attn_norm)) if step_attn_norm else np.nan
            row["mean_attn_std"] = float(np.mean(step_std)) if step_std else np.nan

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.dropna(subset=COMBINED_FEATURES + ["is_correct"])
        df["is_failure"] = 1 - df["is_correct"]
        return df

    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return None


def evaluate_dataset(df, dataset_name):
    print("\n" + "=" * 50)
    print(f"--- RUNNING EVALUATION: {dataset_name} ---")
    print(f"Valid Trajectories (N): {len(df)}")
    print(f"Base Accuracy: {(df['is_correct'].mean() * 100):.1f}%")
    print(f"Base Failure Rate: {(df['is_failure'].mean() * 100):.1f}%")

    X_base = df[OUTPUT_FEATURES]
    X_prop = df[COMBINED_FEATURES]
    y = df["is_failure"]

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    base_aucs, prop_aucs, deltas = [], [], []
    wins = 0

    pipeline_base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline_prop = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])

    for train_idx, test_idx in rskf.split(X_base, y):
        pipeline_base.fit(X_base.iloc[train_idx], y.iloc[train_idx])
        preds_base = pipeline_base.predict_proba(X_base.iloc[test_idx])[:, 1]
        auc_base = roc_auc_score(y.iloc[test_idx], preds_base)

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
    mean_base = np.mean(base_aucs)
    mean_prop = np.mean(prop_aucs)
    mean_delta = np.mean(deltas)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    win_rate = (wins / len(deltas)) * 100
    stat, p_value = wilcoxon(deltas, alternative="greater")

    print(f"\n-> Mean Base AUC:     {mean_base:.4f}")
    print(f"-> Mean Proposed AUC: {mean_prop:.4f}")
    print(f"-> Mean AUC Delta:   +{mean_delta:.4f}")
    print(f"-> 95% CI of Delta:  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"-> Fold Win Rate:     {win_rate:.1f}%")
    print(f"-> Wilcoxon p-value:  {p_value:.2e}")

    return {
        "Dataset": dataset_name,
        "N": len(df),
        "Base_AUC": mean_base,
        "Prop_AUC": mean_prop,
        "Delta": mean_delta,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
        "Win_Rate": win_rate,
        "P_Value": p_value,
    }


if __name__ == "__main__":
    results_summary = []

    for ds in DATASETS:
        df = load_and_clean_data(ds["path"])
        if df is not None:
            res = evaluate_dataset(df, ds["name"])
            results_summary.append(res)

    print("\n\n" + "#" * 90)
    print("                          MASTER RESULTS TABLE (FOR PAPER)")
    print("#" * 90)
    print(f"{'Dataset Model':<18} | {'N':<5} | {'Base AUC':<9} | {'Prop AUC':<9} | {'Delta':<8} | {'95% CI':<20} | {'Win %':<6} | {'p-value'}")
    print("-" * 90)
    for r in results_summary:
        ci_str = f"[{r['CI_Lower']:.3f}, {r['CI_Upper']:.3f}]"
        print(f"{r['Dataset']:<18} | {r['N']:<5} | {r['Base_AUC']:<9.4f} | {r['Prop_AUC']:<9.4f} | +{r['Delta']:<7.4f} | {ci_str:<20} | {r['Win_Rate']:<5.1f}% | {r['P_Value']:.2e}")
    print("#" * 90)