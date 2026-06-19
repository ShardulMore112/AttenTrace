import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

# --- CONFIGURATION ---
OUTPUT_FEATURES = ['out_tau', 'step_out_mean']
ATTN_FEATURES = ['attn_tau', 'step_attn_global', 'step_attn_global_norm', 'step_attn_std']
COMBINED_FEATURES = OUTPUT_FEATURES + ATTN_FEATURES

# File paths
GSM8K_PATH = 'phase2_gsm8k_1000_cleaned.json'
MATH500_PATH = 'phase2_math500_results.json'
OUTPUT_DIR = 'figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    """Loads JSON data, averages lists into scalars, and creates the is_failure target."""
    try:
        df = pd.read_json(filepath)
        
        # Drop rows missing the necessary columns
        df = df.dropna(subset=COMBINED_FEATURES + ['is_correct'])
        df['is_failure'] = 1 - df['is_correct']
        
        # CRITICAL FIX: Convert any lists into single scalar numbers (means)
        for col in COMBINED_FEATURES:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else x)
                
        # Drop any NaNs that might have been created if an empty list was averaged
        df = df.dropna(subset=COMBINED_FEATURES)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
def process_correlations(df, dataset_name):
    """Generates a correlation heatmap and saves the raw data to CSV."""
    print(f"Generating Correlation Matrix for {dataset_name}...")
    
    corr = df[COMBINED_FEATURES].corr()
    
    csv_path = os.path.join(OUTPUT_DIR, f'{dataset_name.lower()}_feature_correlation.csv')
    corr.to_csv(csv_path)
    print(f"  -> Saved raw correlations to {csv_path}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title(f'Feature Correlation Matrix ({dataset_name}, n={len(df)})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fig_corr_{dataset_name.lower()}.png'), dpi=300)
    plt.close()

def plot_roc_curves_cv(df, dataset_name):
    """Generates Out-of-Fold ROC curves using a Pipeline to prevent scaling leakage."""
    print(f"Generating Out-of-Fold ROC Curves for {dataset_name}...")
    
    y = df['is_failure'].values
    
    models = {
        'Output Only': (df[OUTPUT_FEATURES].values, 'blue', '--'),
        'Attention Only': (df[ATTN_FEATURES].values, 'orange', '-.'),
        'Combined': (df[COMBINED_FEATURES].values, 'green', '-')
    }
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess')
    
    # Pipeline prevents data leakage by scaling inside the CV folds
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    
    for label, (X_data, color, linestyle) in models.items():
        y_prob_oof = cross_val_predict(clf, X_data, y, cv=5, method='predict_proba')[:, 1]
        
        fpr, tpr, _ = roc_curve(y, y_prob_oof)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2,
                 label=f'{label} (OOF AUC = {roc_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Out-of-Fold ROC Curves ({dataset_name}, n={len(df)})', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fig_roc_{dataset_name.lower()}.png'), dpi=300)
    plt.close()

def plot_logistic_coefficients(df, dataset_name):
    """Extracts and plots Logistic Regression coefficients for feature importance."""
    print(f"Generating Logistic Regression Coefficients for {dataset_name}...")
    
    X = df[COMBINED_FEATURES].values
    y = df['is_failure'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X_scaled, y)
    
    coefs = pd.DataFrame({
        'Feature': COMBINED_FEATURES,
        'Coefficient': clf.coef_[0]
    }).sort_values(by='Coefficient', ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['orange' if feat in ATTN_FEATURES else 'blue' for feat in coefs['Feature']]
    
    sns.barplot(data=coefs, x='Coefficient', y='Feature', hue='Feature', palette=colors, legend=False)
    plt.title(f'Logistic Regression Coefficients ({dataset_name}, n={len(df)})', fontsize=14)
    plt.xlabel('Coefficient Weight (Positive = Predicts Failure)', fontsize=12)
    plt.ylabel('')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fig_coefficients_{dataset_name.lower()}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("--- GENERATING FINAL PAPER FIGURES (STRICT NO-LEAKAGE) ---")
    
    df_gsm8k = load_data(GSM8K_PATH)
    df_math500 = load_data(MATH500_PATH)
    
    if df_gsm8k is not None:
        process_correlations(df_gsm8k, 'GSM8K')
        plot_roc_curves_cv(df_gsm8k, 'GSM8K')
        plot_logistic_coefficients(df_gsm8k, 'GSM8K')
        
    if df_math500 is not None:
        process_correlations(df_math500, 'MATH-500')
        plot_roc_curves_cv(df_math500, 'MATH-500')
        plot_logistic_coefficients(df_math500, 'MATH-500')
        
    print("\n✅ All visualizations generated. Close your IDE and open Overleaf.")