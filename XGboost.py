# -*- coding: utf-8 -*-
"""
Fast XGBoost for PAM50 subtyping
- No mRMR; uses all gene features
- 5-fold CV + 40-iter RandomizedSearchCV
- After search, increase trees to 1000 and refit on full training set
- Unified structure (same as RandomForest / LogisticRegression reports)
"""
import warnings, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform
from sklearn.base import clone
import joblib, os, json
from datetime import datetime

# --------------------
# 0. Reproducibility
# --------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
# 1. File paths
# --------------------
TRAIN_PATH = r'D:\OneDrive\桌面\新建文件夹\train_dataset_normalized_counts_2.csv'
TEST_PATH  = r'D:\OneDrive\桌面\新建文件夹\test_dataset_normalized_counts_2.csv'

# --------------------
# 2. Load & preprocess
# --------------------
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['PAM50_Subtype'])
y_test  = le.transform(test_df['PAM50_Subtype'])
label_names = le.classes_

# Handle numeric conversion and missing values
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train_raw = train_df[gene_cols].apply(pd.to_numeric, errors='coerce')
X_test_raw  = test_df.reindex(columns=gene_cols).apply(pd.to_numeric, errors='coerce')
train_median = X_train_raw.median(axis=0)
X_train = X_train_raw.fillna(train_median)
X_test  = X_test_raw.fillna(train_median)

print('Train shape:', X_train.shape, '\nLabel counts:\n', pd.Series(y_train).value_counts())
print('Test  shape:', X_test.shape,  '\nLabel counts:\n', pd.Series(y_test).value_counts())

# --------------------
# 3. Base XGBoost model
# --------------------
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=len(np.unique(y_train)),
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# --------------------
# 4. RandomizedSearchCV (40 iterations)
# --------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_dist = {
    'n_estimators': randint(300, 900),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 6),
    'gamma': uniform(0, 0.3)
}

print("\n[Fast Search] RandomizedSearchCV running ...")
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=40,
    scoring='balanced_accuracy',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
random_search.fit(X_train, y_train)
print("\nBest params:", random_search.best_params_)
print("Best CV balanced_acc:", random_search.best_score_)

# --------------------
# 5. 5-Fold CV Evaluation
# --------------------
best_model = random_search.best_estimator_
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_xgboost_full_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "cv_folds"), exist_ok=True)
print(f"\nResults directory: {results_dir}")

fold_results, cv_accuracies, cv_f1_scores = [], [], []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\n=== Fold {fold_idx}/5 ===")
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    fold_model = clone(best_model)
    fold_model.fit(X_fold_train, y_fold_train)
    y_fold_pred = fold_model.predict(X_fold_val)

    acc = accuracy_score(y_fold_val, y_fold_pred)
    f1  = f1_score(y_fold_val, y_fold_pred, average='weighted')
    cv_accuracies.append(acc)
    cv_f1_scores.append(f1)
    fold_results.append({'fold': fold_idx, 'accuracy': acc, 'f1_score': f1})
    print(f"Fold {fold_idx} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

    report = pd.DataFrame(classification_report(
        y_fold_val, y_fold_pred, target_names=label_names, output_dict=True, zero_division=0))
    report.T.to_csv(os.path.join(results_dir, "cv_folds", f"fold_{fold_idx}_classification_report.csv"))

    cm = confusion_matrix(y_fold_val, y_fold_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Fold {fold_idx} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cv_folds", f"fold_{fold_idx}_confusion_matrix.png"), dpi=300)
    plt.close()

print(f"\n=== 5-Fold CV Summary ===")
print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print(f"Mean F1-Score: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")

# --------------------
# 6. Retrain on full data (n_estimators=1000)
# --------------------
best_model.set_params(n_estimators=1000)
best_model.fit(X_train, y_train)

# --------------------
# 7. Final Test Evaluation
# --------------------
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average='weighted')
print("\n=== Final Test Results ===")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=label_names))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
plt.figure(figsize=(10,8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Final Test Set Confusion Matrix (XGBoost, full features)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_test_confusion_matrix.png"), dpi=300)
plt.close()

# --------------------
# 8. Save Results (Unified Format)
# --------------------
cv_results_df = pd.DataFrame(fold_results)
cv_results_df.to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

cv_summary = f"""5-Fold Cross-Validation Summary - XGBoost (Full Features)
========================================
Architecture: XGBoost Classifier
========================================
Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}
Mean F1-Score: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}
Individual Fold Results:"""
for r in fold_results:
    cv_summary += f"\n  Fold {r['fold']}: Accuracy={r['accuracy']:.4f}, F1={r['f1_score']:.4f}"
cv_summary += f"""

Final Test Performance:
  Test Accuracy: {test_acc:.4f}
  Test F1-Score: {test_f1:.4f}

Model Details:
  Best Parameters: {random_search.best_params_}
  CV Search Score: {random_search.best_score_:.4f}
"""
with open(os.path.join(results_dir, "cv_summary.txt"), "w", encoding="utf-8") as f:
    f.write(cv_summary)

# Feature importance
imp_series = pd.Series(best_model.feature_importances_, index=gene_cols).sort_values(ascending=False)
imp_series.head(20).to_csv(os.path.join(results_dir, "top20_genes.csv"), header=['importance'])

xgb_info = {
    "model_type": "XGBoost (Full Features)",
    "n_features": len(gene_cols),
    "best_params": random_search.best_params_,
    "cv_search_score": float(random_search.best_score_),
    "final_test_accuracy": float(test_acc),
    "final_test_f1": float(test_f1),
    "top_10_features": {g: float(v) for g, v in imp_series.head(10).items()}
}
with open(os.path.join(results_dir, "xgboost_info.json"), "w", encoding="utf-8") as f:
    json.dump(xgb_info, f, indent=2, ensure_ascii=False)

# Save model and bundle
joblib.dump(best_model, os.path.join(results_dir, "pam50_xgb_full_model.pkl"))
joblib.dump(best_model, "pam50_xgb_full_fast.pkl")

bundle = {
    "model": best_model,
    "genes_order": list(gene_cols),
    "train_median": train_median,
    "label_names": list(label_names)
}
joblib.dump(bundle, os.path.join(results_dir, "xgb_full_bundle.pkl"))
joblib.dump(bundle, "xgb_full_fast_bundle.pkl")

print(f"\n=== All results have been saved ===")
print(f"Results directory: {results_dir}")
print(f"Contains:")
print("  - cv_results.csv / cv_summary.txt")
print("  - final_test_confusion_matrix.png")
print("  - top20_genes.csv / xgboost_info.json")
print("  - pam50_xgb_full_model.pkl / xgb_full_bundle.pkl")
print(f"\nTotal features used: {len(gene_cols)}")
