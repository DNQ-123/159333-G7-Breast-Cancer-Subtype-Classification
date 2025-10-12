# -*- coding: utf-8 -*-
"""
SVM Five-Subtype Classification (Enhanced Version)
——————————————————————————————————
Additions to the original version:
1) Focal weighting (approx. focal loss via two-pass sample weights)
   - base_gamma
   - two-pass mechanism (first obtain training set probabilities, then construct focal weights, finally retrain with weights)
2) Loss Function Parameters
   - Class-specific Boost Factor (specified by class name JSON)
   - rarity_multiplier (rarity amplification based on training set frequency)
3) Distance Loss Thresholds (probability margin thresholds)
   - margin_threshold & margin_weight (weight samples where "p_true - max_other < threshold")
4) Model Class Weights (passed to sklearn SVM's class_weight)
   - balanced / custom JSON / off

Run: python SVM_plus.py
Dependencies: scikit-learn, pandas, numpy, seaborn, matplotlib, h5py, joblib


- SVM enhancements are injected via sample_weight and class_weight.

"""
import warnings, os, json, joblib, math
import numpy as np
import pandas as pd
import h5py
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# 0. Reproducibility
# -------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------------------------------
# 1. Paths (modify as needed)
# -------------------------------------------------------
CSV_PATH = r'G:\Massey\Mammon2\data\wsi_feature_labels.csv'
H5_ROOT  = r'G:\massey\Mammon2'
OUT_DIR  = './outputs_svm'  # Output root directory
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# 1.1 Tunable Parameters & Recommended Ranges (can directly change values)
# -------------------------------------------------------
# —— Focal (recommended 1.0~3.0, commonly 2.0)
ENABLE_FOCAL      = True
# BASE_GAMMA        = 2.0
BASE_GAMMA        = 2.5
FOCAL_TWO_PASS    = True   # True: first fit base model to get training set probabilities, then generate focal weights and retrain

# —— Class Weights (passed to SVM's class_weight)
#   'none' | 'balanced' | 'custom'
CLASS_WEIGHT_MODE = 'balanced'
CLASS_WEIGHT_JSON = ''     # When mode='custom', format like '{"Luminal A":0.7, "HER2-enriched":1.4}'

# —— Class-specific Boost (recommended 1.0~2.0, small increments)
# CLASS_BOOST_JSON  = '{"HER2-enriched":1.3, "Basal-like":1.2}'
CLASS_BOOST_JSON  = '{"HER2-enriched":1.6, "Basal-like":1.4, "Luminal B":1.1}'
# —— Rarity Multiplier (0.0~2.0, start with 0.5)
# RARITY_MULTIPLIER = 0.5
RARITY_MULTIPLIER = 0.8
# —— Distance Thresholds (probability margin), margin 0.1~0.6, weight 0.1~0.6
ENABLE_MARGIN     = True
# MARGIN_THRESHOLD  = 0.20
MARGIN_THRESHOLD  = 0.15
# MARGIN_WEIGHT     = 0.20
MARGIN_WEIGHT     = 0.30

# —— Others: search iterations and CV folds
N_SPLITS_CV       = 5
# RANDOM_SEARCH_ITERS = 40
RANDOM_SEARCH_ITERS = 60

# -------------------------------------------------------
# 2. Reading + Aggregation
# -------------------------------------------------------
def aggregate_h5(path):
    with h5py.File(path, 'r') as f:
        data = f['features'][:]
    return data.mean(axis=0)  # (D,)


def build_Xy(csv_path, h5_root):
    df = pd.read_csv(csv_path)
    keep = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Solid Tissue Normal']
    df = df[df['Label'].isin(keep)].reset_index(drop=True)

    feats, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Reading+Aggregating'):
        rel = os.path.normpath(str(row['File_Path']).strip())
        # Compatible with relative/absolute
        cand = [os.path.join(h5_root, rel), rel]
        fpath = None
        for c in cand:
            if os.path.isfile(c):
                fpath = c
                break
        if fpath is None:
            raise FileNotFoundError(f"Cannot find H5: {cand}")
        feats.append(aggregate_h5(fpath))
        labels.append(row['Label'])
    return np.array(feats), np.array(labels)

print('Reading and aggregating H5 files ...')
X, y = build_Xy(CSV_PATH, H5_ROOT)
print(f'Done! Sample count={X.shape[0]}, Feature dimension={X.shape[1]}')

# -------------------------------------------------------
# 3. 80 / 20 Stratified Split + Feature Standardization (required for SVM)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# SVM requires feature standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
label_names = le.classes_
num_classes = len(label_names)
print(f'Train: {X_train.shape[0]}  Test: {X_test.shape[0]}')
print('Class mapping:', dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------------------------------
# 4. Construct class_weight (for SVM) & sample weights (two-pass)
# -------------------------------------------------------
# 4.1 class_weight for SVM
svm_class_weight = None
if CLASS_WEIGHT_MODE == 'balanced':
    svm_class_weight = 'balanced'
elif CLASS_WEIGHT_MODE == 'custom':
    try:
        cw = json.loads(CLASS_WEIGHT_JSON) if CLASS_WEIGHT_JSON else {}
    except Exception:
        cw = {}
    # Map class names to indices
    svm_class_weight = {le.transform([k])[0]: float(v) for k, v in cw.items() if k in le.classes_}
else:
    svm_class_weight = None

# 4.2 rarity & boost (first based on training set frequency)
train_counts = Counter(y_train_enc)
mean_freq = np.mean([train_counts.get(i, 1) for i in range(num_classes)])
rarity_vec = np.array([max(mean_freq / max(train_counts.get(i, 1), 1), 1.0) for i in range(num_classes)], dtype=float)

try:
    boost_map = json.loads(CLASS_BOOST_JSON) if CLASS_BOOST_JSON else {}
except Exception:
    boost_map = {}
boost_vec = np.array([float(boost_map.get(lbl, 1.0)) for lbl in label_names], dtype=float)

# 4.3 Phase 1: Optional focal prior (using OOF/IS probability estimation)
base_proba = None
if FOCAL_TWO_PASS and ENABLE_FOCAL:
    print("\n[Pass-1] Getting training set probabilities for focal weights ...")
    # Use a medium configuration model for OOF probability estimation
    base_svm = SVC(
        C=1.0, kernel='rbf', gamma='scale', probability=True,
        random_state=RANDOM_STATE, class_weight=svm_class_weight
    )
    # Use cross_val_predict to generate out-of-fold probabilities, avoid leakage
    cv_oof = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    base_proba = cross_val_predict(
        base_svm, X_train, y_train_enc, cv=cv_oof, method='predict_proba', n_jobs=-1, verbose=0
    )
    # If a fold has no class, sklearn may return nan, will handle later

# 4.4 Assemble sample-level weights (for RandomizedSearchCV & subsequent fitting)
sample_weight_train = np.ones(len(y_train_enc), dtype=float)

# (a) class-specific boost
boost_per_sample = boost_vec[y_train_enc]
sample_weight_train *= boost_per_sample

# (b) rarity multiplier (linear form: 1 + r_mult * (rarity - 1))
if RARITY_MULTIPLIER > 0:
    rarity_per_sample = rarity_vec[y_train_enc]
    sample_weight_train *= (1.0 + RARITY_MULTIPLIER * (rarity_per_sample - 1.0))

# (c) focal weighting (based on OOF probability), w_focal = (1 - p_t) ** gamma
if ENABLE_FOCAL:
    if base_proba is None:
        # Without two-stage, approximate based on prior: p_t ≈ class prior
        prior = np.array([train_counts.get(i, 1) for i in range(num_classes)], dtype=float)
        prior /= prior.sum()
        p_t = prior[y_train_enc]
    else:
        p_t = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
    focal_mult = (1.0 - p_t) ** BASE_GAMMA
    sample_weight_train *= focal_mult

# (d) Distance threshold (probability margin)
if ENABLE_MARGIN:
    if base_proba is None:
        # If no probabilities, skip; could also train a small model to get in-sample probabilities
        # Conservative approach: don't apply margin term
        pass
    else:
        pt = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
        tmp = base_proba.copy()
        tmp[np.arange(len(y_train_enc)), y_train_enc] = -np.inf
        max_other = np.max(tmp, axis=1)
        margin = pt - max_other
        shortfall = np.maximum(0.0, MARGIN_THRESHOLD - margin)
        margin_mult = 1.0 + MARGIN_WEIGHT * (shortfall / max(MARGIN_THRESHOLD, 1e-6))
        sample_weight_train *= margin_mult

# (e) Clipping to avoid extreme weights
# sample_weight_train = np.clip(sample_weight_train, 0.05, 50.0)
sample_weight_train = np.clip(sample_weight_train, 0.03, 80.0)

# -------------------------------------------------------
# 5. Model & Parameter Space
# -------------------------------------------------------
svm = SVC(
    probability=True,
    random_state=RANDOM_STATE,
    class_weight=svm_class_weight
)

param_dist = {
    # 'C': [0.01, 0.1, 1, 10, 100],
    'C': [50, 100, 200, 300, 500],
    # 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'kernel': ['linear'],
    # 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    # 'degree': [2, 3, 4],  # Only for poly kernel
    # 'coef0': [0.0, 0.1, 1.0]  # For poly and sigmoid kernels
}

# -------------------------------------------------------
# 6. 5-Fold Random Search within Training Set (if sample_weight enabled, will be automatically sliced and passed in each fold)
# -------------------------------------------------------
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITERS,
    scoring='balanced_accuracy',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("\n[Fast Search] RandomizedSearchCV running ...")
random_search.fit(X_train, y_train_enc, sample_weight=sample_weight_train)
print("\nBest params:", random_search.best_params_)
print("Best CV balanced_acc:", random_search.best_score_)

# -------------------------------------------------------
# 7. Detailed 5-Fold Cross-Validation Evaluation (output per fold, with sample weights)
# -------------------------------------------------------
best_model = random_search.best_estimator_

results_dir = os.path.join(OUT_DIR, 'svm_results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'cv_folds'), exist_ok=True)

fold_results, cv_accuracies, cv_f1_scores = [], [], []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train_enc), 1):
    print(f"\n=== Fold {fold_idx}/{N_SPLITS_CV} ===")
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train_enc[train_idx], y_train_enc[val_idx]

    w_fold = sample_weight_train[train_idx]

    fold_model = clone(best_model)
    fold_model.fit(X_fold_train, y_fold_train, sample_weight=w_fold)

    y_fold_pred = fold_model.predict(X_fold_val)
    fold_acc = accuracy_score(y_fold_val, y_fold_pred)
    fold_f1  = f1_score(y_fold_val, y_fold_pred, average='weighted')

    cv_accuracies.append(fold_acc)
    cv_f1_scores.append(fold_f1)
    fold_results.append({'fold': fold_idx, 'accuracy': fold_acc, 'f1_score': fold_f1})
    print(f"Fold {fold_idx} - Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

    # Classification report
    fold_report = classification_report(
        y_fold_val, y_fold_pred, target_names=label_names,
        output_dict=True, zero_division=0)
    pd.DataFrame(fold_report).transpose().to_csv(
        os.path.join(results_dir, "cv_folds", f"fold_{fold_idx}_classification_report.csv"))

    # Confusion matrix
    fold_cm = confusion_matrix(y_fold_val, y_fold_pred)
    fold_cm_df = pd.DataFrame(fold_cm, index=label_names, columns=label_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(fold_cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Fold {fold_idx} Confusion Matrix')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cv_folds', f'fold_{fold_idx}_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------
# 8. Retrain on Full Training Set + Test Evaluation (with sample weights)
# -------------------------------------------------------
final_model = clone(best_model)
final_model.fit(X_train, y_train_enc, sample_weight=sample_weight_train)

# Save model and scaler
joblib.dump(final_model, os.path.join(results_dir, 'pam50_svm_model.pkl'))
joblib.dump(le,          os.path.join(results_dir, 'label_encoder.pkl'))
joblib.dump(scaler,      os.path.join(results_dir, 'standard_scaler.pkl'))
# Compatible name
joblib.dump(final_model, os.path.join(OUT_DIR, 'svm_best.pkl'))

# Test set
y_pred_test = final_model.predict(X_test)
acc_test = accuracy_score(y_test_enc, y_pred_test)
f1w_test  = f1_score(y_test_enc, y_pred_test, average='weighted')
print(f"\n=== Final Test ===\nAccuracy={acc_test:.4f}  F1(w)={f1w_test:.4f}")

# Classification report
report = classification_report(y_test_enc, y_pred_test, target_names=label_names,
                               output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv(os.path.join(results_dir, 'final_test_classification_report.csv'))

# Confusion matrix
cm = confusion_matrix(y_test_enc, y_pred_test)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Final Test Set Confusion Matrix (SVM)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'final_test_confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Compatibility old plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (SVM)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=220, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 9. Feature Importance (for linear SVM) and Branch Information
# -------------------------------------------------------
# For linear SVM, we can extract feature importance from coefficients
if hasattr(final_model, 'coef_') and final_model.kernel == 'linear':
    # For multi-class SVM, coef_ is of shape (n_classes, n_features)
    # We take the mean absolute value across classes as feature importance
    coef_abs = np.abs(final_model.coef_).mean(axis=0)
    idx_top = np.argsort(coef_abs)[::-1][:20]
    top_df = pd.DataFrame({
        'rank': np.arange(1, 21),
        'feature_index': idx_top,
        'abs_coef': coef_abs[idx_top]
    })
    top_df.to_csv(os.path.join(results_dir, "top20_features.csv"), index=False)
    print("Top 20 features saved for linear SVM")

# Save branch information (for compatibility with multi-branch fusion pipeline)
branch_info = {
    'model': 'SVM(+focal/boost/rarity/margin/weights)',
    'class_names': list(map(str, label_names)),
    'random_state': RANDOM_STATE,
    'class_weight_mode': CLASS_WEIGHT_MODE,
    'class_weight_json': CLASS_WEIGHT_JSON,
    'enable_focal': ENABLE_FOCAL,
    'base_gamma': BASE_GAMMA,
    'focal_two_pass': FOCAL_TWO_PASS,
    'class_boost_json': CLASS_BOOST_JSON,
    'rarity_multiplier': RARITY_MULTIPLIER,
    'enable_margin': ENABLE_MARGIN,
    'margin_threshold': MARGIN_THRESHOLD,
    'margin_weight': MARGIN_WEIGHT,
    'n_splits_cv': N_SPLITS_CV,
    'random_search_iters': RANDOM_SEARCH_ITERS,
    'best_params': random_search.best_params_
}
with open(os.path.join(results_dir, 'svm_branch_info.json'), 'w', encoding='utf-8') as f:
    json.dump(branch_info, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------
# 10. Result File Summary
# -------------------------------------------------------
pd.DataFrame(fold_results).to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

cv_summary_content = f"""5-Fold Cross-Validation Summary - SVM(+enhanced)
========================================
Architecture: SVM Classifier
========================================
Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}
Mean F1(w):   {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}
Best Params:  {random_search.best_params_}
Class Weight Mode: {CLASS_WEIGHT_MODE}
Focal: {ENABLE_FOCAL} (gamma={BASE_GAMMA}, two-pass={FOCAL_TWO_PASS})
Boost: {CLASS_BOOST_JSON}
Rarity Multiplier: {RARITY_MULTIPLIER}
Margin: {ENABLE_MARGIN} (thr={MARGIN_THRESHOLD}, w={MARGIN_WEIGHT})
"""
with open(os.path.join(results_dir, 'cv_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(cv_summary_content)

# Training process metrics (compatible with downstream)
training_metrics_df = pd.DataFrame([
    {'epoch': i, 'train_loss': 'N/A', 'train_acc': 'N/A', 'val_loss': 'N/A', 'val_acc': r['accuracy'], 'val_f1': r['f1_score']}
    for i, r in enumerate(fold_results, 1)
])
training_metrics_df.to_csv(os.path.join(results_dir, 'final_training_metrics.csv'), index=False)

print(f"\n=== All results saved ===")
print(f"Result directory: {results_dir}")
print(f"Included files:")
print(f"  - cv_results.csv / cv_summary.txt")
print(f"  - final_test_classification_report.csv / final_test_confusion_matrix.png")
print(f"  - cv_folds/  detailed reports and confusion matrices for each fold")
print(f"  - svm_branch_info.json")
if hasattr(final_model, 'coef_') and final_model.kernel == 'linear':
    print(f"  - top20_features.csv (for linear SVM)")
print(f"  - pam50_svm_model.pkl / label_encoder.pkl / standard_scaler.pkl")
print(f"Root directory compatible files: svm_best.pkl / confusion_matrix.png")