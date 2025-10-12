# -*- coding: utf-8 -*-
"""
XGBoost 五亚型分类（增强版）
在原 RandomForest 基础上：
1) 模型替换为 XGBoost
2) 保留 focal / boost / rarity / margin 样本权重
3) 使用 scale_pos_weight 处理类别不平衡
4) 其余流程不变：RandomizedSearchCV、5 折 CV、测试集评估、结果保存
运行： python xgboost_plus.py
依赖：scikit-learn, pandas, numpy, seaborn, matplotlib, h5py, joblib, xgboost
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)
from tqdm import tqdm
import xgboost as xgb

warnings.filterwarnings("ignore")

print(xgb.__version__)        # 应 ≥1.6.0
print(xgb.build_info()) 

# -------------------------------------------------------
# 0. Reproducibility
# -------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------------------------------
# 1. 路径（按需修改）
# -------------------------------------------------------
CSV_PATH = r'G:\Massey\Mammon2\data\wsi_feature_labels.csv'
H5_ROOT  = r'G:\massey\Mammon2'
OUT_DIR  = './outputs_xgb'  # 输出根目录
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# 1.1 核心参数（沿用上一轮最优）
# -------------------------------------------------------
# --- 模型选择 ---
MODEL_NAME = 'XGBoost'

# -------------------------------------------------------
# 1.1 可调参数 & 建议范围（可直接改数值）
# -------------------------------------------------------
# —— Focal（建议 1.0~3.0，常用 2.0）
ENABLE_FOCAL      = False
BASE_GAMMA        = 3.5    #Focal Loss的gamma值，越大越关注难样本
FOCAL_TWO_PASS    = False   # True: 先拟合基础模型获取训练集概率，再以此生成 focal 权重并重训

# —— Class Weights（传给 RF 的 class_weight）
#   'none' | 'balanced' | 'balanced_subsample' | 'custom'
CLASS_WEIGHT_MODE = 'balanced'
CLASS_WEIGHT_JSON = ''     # 当 mode='custom' 时，形如 '{"Luminal A":0.7, "HER2-enriched":1.4}'

# —— Class-specific Boost（建议 1.0~2.0，小步增）
CLASS_BOOST_JSON = '{"HER2-enriched":1.5, "Basal-like":1.3, "Luminal B":1.2}'
# —— Rarity Multiplier（0.0~2.0，起步 0.5）
RARITY_MULTIPLIER = 1.2

# —— 距离阈值（概率边距），margin 0.1~0.6，权重 0.1~0.6
ENABLE_MARGIN     = True
MARGIN_THRESHOLD  = 0.1   #概率边距阈值，越小越严格
MARGIN_WEIGHT     = 0.50   #边距惩罚强度

# —— 其他：搜索迭代与CV折数
N_SPLITS_CV       = 5
RANDOM_SEARCH_ITERS = 40

# -------------------------------------------------------
# 2. 读取+聚合（复用原函数）
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc='读取+聚合'):
        rel = os.path.normpath(str(row['File_Path']).strip())
        # 兼容相对/绝对
        cand = [os.path.join(h5_root, rel), rel]
        fpath = None
        for c in cand:
            if os.path.isfile(c):
                fpath = c
                break
        if fpath is None:
            raise FileNotFoundError(f"找不到H5: {cand}")
        feats.append(aggregate_h5(fpath))
        labels.append(row['Label'])
    return np.array(feats), np.array(labels)

print('正在读取并聚合 H5 文件 …')
X, y = build_Xy(CSV_PATH, H5_ROOT)
print(f'完成！样本数={X.shape[0]}, 特征维数={X.shape[1]}')

# -------------------------------------------------------
# 3. 80 / 20 分层拆分
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
label_names = le.classes_
num_classes = len(label_names)
print(f'Train: {X_train.shape[0]}  Test: {X_test.shape[0]}')
print('类别映射:', dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------------------------------
# 4. 构造样本权重
# -------------------------------------------------------
train_counts = Counter(y_train_enc)
mean_freq = np.mean([train_counts.get(i, 1) for i in range(num_classes)])
rarity_vec = np.array([max(mean_freq / max(train_counts.get(i, 1), 1), 1.0) for i in range(num_classes)], dtype=float)

boost_map = json.loads(CLASS_BOOST_JSON) if CLASS_BOOST_JSON else {}
boost_vec = np.array([float(boost_map.get(lbl, 1.0)) for lbl in label_names], dtype=float)

# --- 获取训练集概率（two-pass focal） ---
base_proba = None
if FOCAL_TWO_PASS and ENABLE_FOCAL:
    print("\n[Pass-1] 获取训练集概率用于 focal 权重 …")
    base_rf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    cv_oof = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    base_proba = cross_val_predict(
        base_rf, X_train, y_train_enc, cv=cv_oof, method='predict_proba', n_jobs=-1, verbose=0
    )

# --- 组装样本权重 ---
sample_weight_train = np.ones(len(y_train_enc), dtype=float)
sample_weight_train *= boost_vec[y_train_enc]
if RARITY_MULTIPLIER > 0:
    rarity_per_sample = rarity_vec[y_train_enc]
    sample_weight_train *= (1.0 + RARITY_MULTIPLIER * (rarity_per_sample - 1.0))

if ENABLE_FOCAL:
    if base_proba is None:
        prior = np.array([train_counts.get(i, 1) for i in range(num_classes)], dtype=float)
        prior /= prior.sum()
        p_t = prior[y_train_enc]
    else:
        p_t = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
    focal_mult = (1.0 - p_t) ** BASE_GAMMA
    sample_weight_train *= focal_mult

if ENABLE_MARGIN and base_proba is not None:
    pt = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
    tmp = base_proba.copy()
    tmp[np.arange(len(y_train_enc)), y_train_enc] = -np.inf
    max_other = np.max(tmp, axis=1)
    margin = pt - max_other
    shortfall = np.maximum(0.0, MARGIN_THRESHOLD - margin)
    margin_mult = 1.0 + MARGIN_WEIGHT * (shortfall / max(MARGIN_THRESHOLD, 1e-6))
    sample_weight_train *= margin_mult

sample_weight_train = np.clip(sample_weight_train, 0.05, 30.0)

# -------------------------------------------------------
# 5. XGBoost 参数空间
# -------------------------------------------------------
xgb_cls = xgb.XGBClassifier(
    n_estimators=1000,
    random_state=RANDOM_STATE,
    eval_metric='mlogloss',
    tree_method='gpu_hist',      # ← 1. 用 GPU 建树
    predictor='gpu_predictor',   # ← 2. 预测也放 GPU
    n_jobs=1                     # ← 3. GPU 下 cpu 线程设 0 最快
)

param_dist = {
    'n_estimators': [400,800],          # 控制树数量，太多训练时间长
    'max_depth': [4, 8],                 # 较小深度能减少训练量
    'learning_rate': [0.06, 0.08],       # 保持学习率适中
    'subsample': [0.8],                  # 固定，减少组合
    'colsample_bytree': [0.8],           # 固定，性能稳定
    'gamma': [0, 0.1],                   # 控制分裂复杂度
    'reg_alpha': [0, 0.1],               # L1 正则化
    'reg_lambda': [0.8, 1.5],            # L2 正则化
    'min_child_weight': [1, 3],          # 控制过拟合
    'tree_method': ['gpu_hist'],
    'predictor': ['gpu_predictor'],
}

# -------------------------------------------------------
# 6. RandomizedSearchCV（带样本权重）
# -------------------------------------------------------
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    estimator=xgb_cls,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITERS,
    scoring='balanced_accuracy',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("\n[Fast Search] RandomizedSearchCV running …")
random_search.fit(X_train, y_train_enc, sample_weight=sample_weight_train)
print("\nBest params:", random_search.best_params_)
print("Best CV balanced_acc:", random_search.best_score_)

# -------------------------------------------------------
# 7. 5 折 CV 逐折评估（同原逻辑）
# -------------------------------------------------------
best_model = random_search.best_estimator_

results_dir = os.path.join(OUT_DIR, 'xgb_results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'cv_folds'), exist_ok=True)

fold_results, cv_accuracies, cv_f1_scores = [], [], []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train_enc), 1):
    print(f"\n=== Fold {fold_idx}/{N_SPLITS_CV} ===")
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train_enc[train_idx], y_train_enc[val_idx]
    w_fold = sample_weight_train[train_idx]

    fold_model = best_model.__class__(**best_model.get_params())
    fold_model.fit(X_fold_train, y_fold_train,
                   sample_weight=w_fold,
                   verbose=False)

    y_fold_pred = fold_model.predict(X_fold_val)
    fold_acc = accuracy_score(y_fold_val, y_fold_pred)
    fold_f1 = f1_score(y_fold_val, y_fold_pred, average='weighted')

    cv_accuracies.append(fold_acc); cv_f1_scores.append(fold_f1)
    fold_results.append({'fold': fold_idx, 'accuracy': fold_acc, 'f1_score': fold_f1})
    print(f"Fold {fold_idx} - Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

    # 分类报告
    fold_report = classification_report(
        y_fold_val, y_fold_pred, target_names=label_names,
        output_dict=True, zero_division=0)
    pd.DataFrame(fold_report).transpose().to_csv(
        os.path.join(results_dir, "cv_folds", f"fold_{fold_idx}_classification_report.csv"))

    # 混淆矩阵
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
# 8. 全训练集重训 + 测试评估
# -------------------------------------------------------
final_model = best_model.__class__(**best_model.get_params())
final_model.fit(X_train, y_train_enc,
                sample_weight=sample_weight_train,
                verbose=False)

# 保存模型
joblib.dump(final_model, os.path.join(results_dir, 'pam50_xgb_model.pkl'))
joblib.dump(le,          os.path.join(results_dir, 'label_encoder.pkl'))
joblib.dump(final_model, os.path.join(OUT_DIR, 'xgb_best.pkl'))

# 测试集
y_pred_test = final_model.predict(X_test)
acc_test = accuracy_score(y_test_enc, y_pred_test)
f1w_test = f1_score(y_test_enc, y_pred_test, average='weighted')
print(f"\n=== Final Test ===\nAccuracy={acc_test:.4f}  F1(w)={f1w_test:.4f}")

# 分类报告
report = classification_report(y_test_enc, y_pred_test, target_names=label_names,
                               output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv(os.path.join(results_dir, 'final_test_classification_report.csv'))

# 混淆矩阵
cm = confusion_matrix(y_test_enc, y_pred_test)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Final Test Set Confusion Matrix (XGBoost)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'final_test_confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 兼容旧图
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (XGBoost)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=220, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 9. 特征重要性（XGB 原生）
# -------------------------------------------------------
importances = final_model.feature_importances_
idx_top = np.argsort(importances)[::-1][:20]
top_df = pd.DataFrame({
    'rank': np.arange(1, len(idx_top)+1),
    'feature_index': idx_top,
    'importance': importances[idx_top]
})
top_df.to_csv(os.path.join(results_dir, 'top20_features.csv'), index=False)

# -------------------------------------------------------
# 10. 分支信息
# -------------------------------------------------------
branch_info = {
    'model': 'XGBoost(+focal/boost/rarity/margin/sample_weight)',
    'class_names': list(map(str, label_names)),
    'random_state': RANDOM_STATE,
    'best_params': random_search.best_params_,
    'class_weight_mode': CLASS_WEIGHT_MODE,
    'enable_focal': ENABLE_FOCAL,
    'base_gamma': BASE_GAMMA,
    'focal_two_pass': FOCAL_TWO_PASS,
    'class_boost_json': CLASS_BOOST_JSON,
    'rarity_multiplier': RARITY_MULTIPLIER,
    'enable_margin': ENABLE_MARGIN,
    'margin_threshold': MARGIN_THRESHOLD,
    'margin_weight': MARGIN_WEIGHT,
    'n_splits_cv': N_SPLITS_CV,
    'random_search_iters': RANDOM_SEARCH_ITERS
}
with open(os.path.join(results_dir, 'xgb_branch_info.json'), 'w', encoding='utf-8') as f:
    json.dump(branch_info, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------
# 11. 结果汇总
# -------------------------------------------------------
pd.DataFrame(fold_results).to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

cv_summary_content = f"""5-Fold Cross-Validation Summary - XGBoost(+enhanced)
========================================
Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}
Mean F1(w):   {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}
Best Params:  {random_search.best_params_}
Test Accuracy: {acc_test:.4f}
Test F1(w): {f1w_test:.4f}
"""
with open(os.path.join(results_dir, 'cv_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(cv_summary_content)

print(f"\n=== 所有结果已保存 ===")
print(f"结果目录: {results_dir}")