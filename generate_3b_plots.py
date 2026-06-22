import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load the 3B Data
filepath = "data/math500/phase2_math500_3B_graded.json"
with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for record in data:
    if record.get("is_correct") is None: continue
    row = {"id": record["id"], "is_correct": record["is_correct"]}
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

df = pd.DataFrame(rows).dropna()
df["is_failure"] = 1 - df["is_correct"]

OUTPUT_FEATURES = ["out_tau", "mean_out_entropy", "max_out_entropy"]
ATTN_FEATURES = ["attn_tau", "mean_attn_entropy", "max_attn_entropy", "mean_attn_std"]
COMBINED_FEATURES = OUTPUT_FEATURES + ATTN_FEATURES

y = df["is_failure"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced"))])

# --- PLOT 1: ROC CURVES ---
plt.figure(figsize=(8, 8), dpi=300)
for features, name, color, style in [
    (OUTPUT_FEATURES, "Output Only", "blue", "--"),
    (ATTN_FEATURES, "Attention Only", "orange", "-."),
    (COMBINED_FEATURES, "Combined", "green", "-")
]:
    preds = cross_val_predict(pipeline, df[features], y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, preds)
    score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linestyle=style, lw=2, label=f"{name} (OOF AUC = {score:.3f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Guess")
plt.title("Out-of-Fold ROC Curves (MATH-500 3B, n=254)", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.savefig("fig_roc_math-500_3B.png", bbox_inches="tight")
plt.close()

# --- PLOT 2: COEFFICIENTS ---
pipeline.fit(df[COMBINED_FEATURES], y)
coefs = pipeline.named_steps["lr"].coef_[0]
plt.figure(figsize=(10, 6), dpi=300)
colors = ['blue' if c in OUTPUT_FEATURES else 'goldenrod' for c in COMBINED_FEATURES]
plt.barh(COMBINED_FEATURES, coefs, color=colors)
plt.axvline(0, color='black', lw=1)
plt.title("Logistic Regression Coefficients (MATH-500 3B, n=254)", fontsize=14)
plt.xlabel("Coefficient Weight (Positive = Predicts Failure)", fontsize=12)
plt.savefig("fig_coefficients_math-500_3B.png", bbox_inches="tight")
plt.close()

print("Successfully generated fig_roc_math-500_3B.png and fig_coefficients_math-500_3B.png!")