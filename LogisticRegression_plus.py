# -*- coding: utf-8 -*-
"""
LogisticRegression 五亚型分类（增强版 v2）
——————————————————————————————————
改进要点：
1) 训练集:测试集 = 8:2（各类别保持比例）
2) 训练集内部进行 5-Fold 交叉验证（不变）
3) 调整 focal / boost / rarity / margin 参数
4) 扩大 RandomizedSearchCV 搜索空间与迭代次数
5) 提高 max_iter、增加收敛稳定性
"""

import warnings, os, json, joblib
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# 0. Reproducibility
# -------------------------------------------------------
RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# -------------------------------------------------------
# 1. 路径与参数设置
# -------------------------------------------------------
CSV_PATH = r"E:\Massey\Mammon2\wsi_feature_labels.csv"
H5_ROOT  = r"E:\Massey\Mammon2"
OUT_DIR  = r"./outputs_lr_plus_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Focal & Boost 系列参数
ENABLE_FOCAL      = True
BASE_GAMMA        = 1.5
FOCAL_TWO_PASS    = True

CLASS_WEIGHT_MODE = 'balanced'  # 'none'|'balanced'|'custom'
CLASS_BOOST_JSON  = '{"HER2-enriched":1.5, "Basal-like":1.3, "Solid Tissue Normal":1.2}'
RARITY_MULTIPLIER = 0.8

ENABLE_MARGIN     = True
MARGIN_THRESHOLD  = 0.25
MARGIN_WEIGHT     = 0.35

# 搜索参数
N_SPLITS_CV        = 5
RANDOM_SEARCH_ITERS = 60

# -------------------------------------------------------
# 2. 数据加载与聚合
# -------------------------------------------------------
def load_h5_features(path):
    with h5py.File(path, 'r') as f:
        data = f['features'][:]
    return data.mean(axis=0)

def build_Xy(csv_path, h5_root):
    df = pd.read_csv(csv_path)
    keep = ['Basal-like','HER2-enriched','Luminal A','Luminal B','Solid Tissue Normal']
    df = df[df['Label'].isin(keep)].reset_index(drop=True)
    feats, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="读取+聚合 H5"):
        path = os.path.join(h5_root, row["File_Path"])
        if not os.path.isfile(path):
            print(f"⚠️ 缺失文件: {path}")
            continue
        feats.append(load_h5_features(path))
        labels.append(row["Label"])
    return np.array(feats), np.array(labels)

print("📥 正在加载特征 …")
X, y = build_Xy(CSV_PATH, H5_ROOT)
print(f"✅ 完成! 样本数={X.shape[0]}, 特征维数={X.shape[1]}")

# -------------------------------------------------------
# 3. 拆分与标准化 (按类别 8:2)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
label_names = le.classes_
num_classes = len(label_names)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train={X_train.shape[0]}  Test={X_test.shape[0]}")
print("类别映射:", dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------------------------------
# 4. 权重构造
# -------------------------------------------------------
train_counts = Counter(y_train_enc)
mean_freq = np.mean(list(train_counts.values()))
rarity_vec = np.array([max(mean_freq / train_counts.get(i,1), 1.0) for i in range(num_classes)], float)

try:
    boost_map = json.loads(CLASS_BOOST_JSON) if CLASS_BOOST_JSON else {}
except Exception: boost_map = {}
boost_vec = np.array([float(boost_map.get(lbl, 1.0)) for lbl in label_names], float)

sample_weight_train = np.ones(len(y_train_enc), float)

# Boost
sample_weight_train *= boost_vec[y_train_enc]
# Rarity
sample_weight_train *= (1.0 + RARITY_MULTIPLIER * (rarity_vec[y_train_enc] - 1.0))

# Focal & Margin 前向概率
base_proba = None
if FOCAL_TWO_PASS and ENABLE_FOCAL:
    print("\n[Pass-1] Logistic 基础概率估计 …")
    base_lr = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=800, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    cv_pred = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    base_proba = cross_val_predict(base_lr, X_train, y_train_enc, cv=cv_pred,
                                   method="predict_proba", n_jobs=-1, verbose=0)

# Focal weighting
if ENABLE_FOCAL:
    if base_proba is None:
        prior = np.array([train_counts.get(i, 1) for i in range(num_classes)], float)
        prior /= prior.sum()
        p_t = prior[y_train_enc]
    else:
        p_t = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
    sample_weight_train *= (1.0 - p_t) ** BASE_GAMMA

# Margin weighting
if ENABLE_MARGIN and base_proba is not None:
    pt = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
    tmp = base_proba.copy()
    tmp[np.arange(len(y_train_enc)), y_train_enc] = -np.inf
    max_other = np.max(tmp, axis=1)
    margin = pt - max_other
    shortfall = np.maximum(0.0, MARGIN_THRESHOLD - margin)
    margin_mult = 1.0 + MARGIN_WEIGHT * (shortfall / max(MARGIN_THRESHOLD, 1e-6))
    sample_weight_train *= margin_mult

sample_weight_train = np.clip(sample_weight_train, 0.05, 50.0)

# -------------------------------------------------------
# 5. 调参空间与搜索
# -------------------------------------------------------
param_dist = {
    'C': loguniform(1e-4, 1e4),
    'solver': ['lbfgs', 'newton-cg', 'saga', 'sag'],
    'penalty': ['l2', 'none']
}

base_lr = LogisticRegression(
    multi_class="multinomial",
    class_weight=CLASS_WEIGHT_MODE if CLASS_WEIGHT_MODE != 'none' else None,
    max_iter=4000,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

print("\n[Fast Search] RandomizedSearchCV running …")
search = RandomizedSearchCV(
    estimator=base_lr,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITERS,
    scoring='f1_macro',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
search.fit(X_train, y_train_enc, sample_weight=sample_weight_train)
print("\nBest params:", search.best_params_)
print("Best CV f1_macro:", search.best_score_)

# -------------------------------------------------------
# 6. 五折交叉验证评估
# -------------------------------------------------------
best_model = search.best_estimator_
results_dir = os.path.join(OUT_DIR, "lr_results")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "cv_folds"), exist_ok=True)

fold_results, cv_accs, cv_f1s = [], [], []

for i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train_enc), 1):
    print(f"\n=== Fold {i}/{N_SPLITS_CV} ===")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train_enc[tr_idx], y_train_enc[val_idx]
    w_tr = sample_weight_train[tr_idx]

    model = clone(best_model)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1w = f1_score(y_val, y_pred, average="weighted")
    cv_accs.append(acc); cv_f1s.append(f1w)
    fold_results.append({'fold': i, 'accuracy': acc, 'f1_weighted': f1w})
    print(f"Fold {i} - Acc={acc:.4f}, F1w={f1w:.4f}")

    rep = classification_report(y_val, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(results_dir, "cv_folds", f"fold_{i}_classification_report.csv"))
    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Purples')
    plt.title(f"Fold {i} Confusion Matrix (LR)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cv_folds", f"fold_{i}_confusion_matrix.png"), dpi=300)
    plt.close()

# -------------------------------------------------------
# 7. 全训练集重训 + 测试评估
# -------------------------------------------------------
final_model = clone(best_model)
final_model.fit(X_train, y_train_enc, sample_weight=sample_weight_train)

joblib.dump(final_model, os.path.join(results_dir, "pam50_lr_model.pkl"))
joblib.dump(le, os.path.join(results_dir, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(results_dir, "scaler.pkl"))
joblib.dump(final_model, os.path.join(OUT_DIR, "lr_best.pkl"))

y_pred_test = final_model.predict(X_test)
acc_test = accuracy_score(y_test_enc, y_pred_test)
f1w_test = f1_score(y_test_enc, y_pred_test, average='weighted')
print(f"\n=== Final Test ===\nAcc={acc_test:.4f}  F1w={f1w_test:.4f}")

rep_test = classification_report(y_test_enc, y_pred_test, target_names=label_names, output_dict=True, zero_division=0)
pd.DataFrame(rep_test).transpose().to_csv(os.path.join(results_dir, 'final_test_classification_report.csv'))

cm = confusion_matrix(y_test_enc, y_pred_test)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Final Test Confusion Matrix (LR)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_test_confusion_matrix.png"), dpi=300)
plt.close()

# -------------------------------------------------------
# 8. 系数重要性与元信息
# -------------------------------------------------------
coef_abs = np.abs(final_model.coef_).mean(axis=0)
idx_top = np.argsort(coef_abs)[::-1][:20]
top_df = pd.DataFrame({
    'rank': np.arange(1, 21),
    'feature_index': idx_top,
    'abs_coef': coef_abs[idx_top]
})
top_df.to_csv(os.path.join(results_dir, "top20_features.csv"), index=False)

branch_info = {
    'model': 'LogisticRegression(+focal/boost/rarity/margin)',
    'class_names': list(label_names),
    'random_state': RANDOM_STATE,
    'class_weight_mode': CLASS_WEIGHT_MODE,
    'focal': ENABLE_FOCAL,
    'base_gamma': BASE_GAMMA,
    'boost': CLASS_BOOST_JSON,
    'rarity': RARITY_MULTIPLIER,
    'margin': ENABLE_MARGIN,
    'margin_thr': MARGIN_THRESHOLD,
    'margin_w': MARGIN_WEIGHT,
    'cv_splits': N_SPLITS_CV,
    'random_search_iters': RANDOM_SEARCH_ITERS,
    'best_params': search.best_params_
}
with open(os.path.join(results_dir, "lr_branch_info.json"), "w", encoding="utf-8") as f:
    json.dump(branch_info, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------
# 9. 结果汇总
# -------------------------------------------------------
pd.DataFrame(fold_results).to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

summary = f"""5-Fold Cross-Validation Summary - LogisticRegression(+enhanced)
========================================
Architecture: Logistic Regression (multinomial)
========================================
Mean Accuracy: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}
Mean F1(w):   {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}
Best Params:  {search.best_params_}
Class Weight Mode: {CLASS_WEIGHT_MODE}
Focal: {ENABLE_FOCAL} (γ={BASE_GAMMA}, two-pass={FOCAL_TWO_PASS})
Boost: {CLASS_BOOST_JSON}
Rarity Multiplier: {RARITY_MULTIPLIER}
Margin: {ENABLE_MARGIN} (thr={MARGIN_THRESHOLD}, w={MARGIN_WEIGHT})
"""
with open(os.path.join(results_dir, "cv_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary)

metrics_df = pd.DataFrame([
    {'epoch': i, 'val_acc': r['accuracy'], 'val_f1': r['f1_weighted']}
    for i, r in enumerate(fold_results, 1)
])
metrics_df.to_csv(os.path.join(results_dir, "final_training_metrics.csv"), index=False)

print("\n=== 所有结果已保存 ===")
print(f"结果目录: {results_dir}")
print(f"包含文件:")
print("  - cv_results.csv / cv_summary.txt")
print("  - final_test_classification_report.csv / final_test_confusion_matrix.png")
print("  - cv_folds/ 每折详细报告与混淆矩阵")
print("  - lr_branch_info.json / top20_features.csv")
print("  - pam50_lr_model.pkl / label_encoder.pkl / scaler.pkl")
print("根目录兼容文件: lr_best.pkl / confusion_matrix.png")