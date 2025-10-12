# -*- coding: utf-8 -*-
"""
SVM 五亚型分类（增强版）
——————————————————————————————————
在原版基础上增加：
1) Focal weighting（approx. focal loss via two-pass sample weights）
   - base_gamma
   - two-pass 机制（先获得训练集概率，再构造 focal 权重，最后带权重重新训练）
2) Loss Function Parameters
   - Class-specific Boost Factor（按类别名 JSON 指定）
   - rarity_multiplier（基于训练集频次的稀有度放大）
3) Distance Loss Thresholds（概率边距阈值）
   - margin_threshold & margin_weight（对“p_true - max_other < 阈值”的样本加权）
4) Model Class Weights（传递给 sklearn SVM 的 class_weight）
   - balanced / 自定义 JSON / 关闭

运行： python SVM_plus.py
依赖：scikit-learn, pandas, numpy, seaborn, matplotlib, h5py, joblib


- SVM 通过 sample_weight 与 class_weight 注入增强机制。

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
# 1. 路径（按需修改）
# -------------------------------------------------------
CSV_PATH = r'G:\Massey\Mammon2\data\wsi_feature_labels.csv'
H5_ROOT  = r'G:\massey\Mammon2'
OUT_DIR  = './outputs_svm'  # 输出根目录
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# 1.1 可调参数 & 建议范围（可直接改数值）
# -------------------------------------------------------
# —— Focal（建议 1.0~3.0，常用 2.0）
ENABLE_FOCAL      = True
# BASE_GAMMA        = 2.0
BASE_GAMMA        = 2.5
FOCAL_TWO_PASS    = True   # True: 先拟合基础模型获取训练集概率，再以此生成 focal 权重并重训

# —— Class Weights（传给 SVM 的 class_weight）
#   'none' | 'balanced' | 'custom'
CLASS_WEIGHT_MODE = 'balanced'
CLASS_WEIGHT_JSON = ''     # 当 mode='custom' 时，形如 '{"Luminal A":0.7, "HER2-enriched":1.4}'

# —— Class-specific Boost（建议 1.0~2.0，小步增）
# CLASS_BOOST_JSON  = '{"HER2-enriched":1.3, "Basal-like":1.2}'
CLASS_BOOST_JSON  = '{"HER2-enriched":1.6, "Basal-like":1.4, "Luminal B":1.1}'
# —— Rarity Multiplier（0.0~2.0，起步 0.5）
# RARITY_MULTIPLIER = 0.5
RARITY_MULTIPLIER = 0.8
# —— 距离阈值（概率边距），margin 0.1~0.6，权重 0.1~0.6
ENABLE_MARGIN     = True
# MARGIN_THRESHOLD  = 0.20
MARGIN_THRESHOLD  = 0.15
# MARGIN_WEIGHT     = 0.20
MARGIN_WEIGHT     = 0.30

# —— 其他：搜索迭代与CV折数
N_SPLITS_CV       = 5
# RANDOM_SEARCH_ITERS = 40
RANDOM_SEARCH_ITERS = 60

# -------------------------------------------------------
# 2. 读取+聚合
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
# 3. 80 / 20 分层拆分 + 特征标准化（SVM必需）
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# SVM需要特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
label_names = le.classes_
num_classes = len(label_names)
print(f'Train: {X_train.shape[0]}  Test: {X_test.shape[0]}')
print('类别映射:', dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------------------------------
# 4. 构建 class_weight（给 SVM） & 样本权重（two-pass）
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
    # 将类名映射到索引
    svm_class_weight = {le.transform([k])[0]: float(v) for k, v in cw.items() if k in le.classes_}
else:
    svm_class_weight = None

# 4.2 rarity & boost（先基于训练集频次）
train_counts = Counter(y_train_enc)
mean_freq = np.mean([train_counts.get(i, 1) for i in range(num_classes)])
rarity_vec = np.array([max(mean_freq / max(train_counts.get(i, 1), 1), 1.0) for i in range(num_classes)], dtype=float)

try:
    boost_map = json.loads(CLASS_BOOST_JSON) if CLASS_BOOST_JSON else {}
except Exception:
    boost_map = {}
boost_vec = np.array([float(boost_map.get(lbl, 1.0)) for lbl in label_names], dtype=float)

# 4.3 第一阶段：可选 focal 先验（用 OOF/IS 概率估计）
base_proba = None
if FOCAL_TWO_PASS and ENABLE_FOCAL:
    print("\n[Pass-1] 获取训练集概率用于 focal 权重 …")
    # 用一个中等配置模型做 OOF 概率估计
    base_svm = SVC(
        C=1.0, kernel='rbf', gamma='scale', probability=True,
        random_state=RANDOM_STATE, class_weight=svm_class_weight
    )
    # 采用 cross_val_predict 产生 out-of-fold 概率，避免泄露
    cv_oof = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    base_proba = cross_val_predict(
        base_svm, X_train, y_train_enc, cv=cv_oof, method='predict_proba', n_jobs=-1, verbose=0
    )
    # 如果某折没有类，sklearn可能返回nan，后续会处理

# 4.4 组装样本级权重（供 RandomizedSearchCV & 后续拟合使用）
sample_weight_train = np.ones(len(y_train_enc), dtype=float)

# (a) class-specific boost
boost_per_sample = boost_vec[y_train_enc]
sample_weight_train *= boost_per_sample

# (b) rarity multiplier（线性形式：1 + r_mult * (rarity - 1)）
if RARITY_MULTIPLIER > 0:
    rarity_per_sample = rarity_vec[y_train_enc]
    sample_weight_train *= (1.0 + RARITY_MULTIPLIER * (rarity_per_sample - 1.0))

# (c) focal weighting（基于 OOF 概率），w_focal = (1 - p_t) ** gamma
if ENABLE_FOCAL:
    if base_proba is None:
        # 不做两阶段，则基于先验近似：p_t ≈ 类先验
        prior = np.array([train_counts.get(i, 1) for i in range(num_classes)], dtype=float)
        prior /= prior.sum()
        p_t = prior[y_train_enc]
    else:
        p_t = np.clip(base_proba[np.arange(len(y_train_enc)), y_train_enc], 1e-6, 1-1e-6)
    focal_mult = (1.0 - p_t) ** BASE_GAMMA
    sample_weight_train *= focal_mult

# (d) 距离阈值（概率边距）
if ENABLE_MARGIN:
    if base_proba is None:
        # 若无概率，跳过；也可改为先训练一个小模型拿 in-sample 概率
        # 这里保守处理：不应用边距项
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

# (e) 裁剪，避免极端权重
# sample_weight_train = np.clip(sample_weight_train, 0.05, 50.0)
sample_weight_train = np.clip(sample_weight_train, 0.03, 80.0)

# -------------------------------------------------------
# 5. 模型 & 参数空间
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
    # 'degree': [2, 3, 4],  # 仅用于poly核
    # 'coef0': [0.0, 0.1, 1.0]  # 用于poly和sigmoid核
}

# -------------------------------------------------------
# 6. 训练集内 5 折随机搜索（若启用 sample_weight，会在各折自动切片传入）
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

print("\n[Fast Search] RandomizedSearchCV running …")
random_search.fit(X_train, y_train_enc, sample_weight=sample_weight_train)
print("\nBest params:", random_search.best_params_)
print("Best CV balanced_acc:", random_search.best_score_)

# -------------------------------------------------------
# 7. 详细的 5 折交叉验证评估（逐折输出，携带样本权重）
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
# 8. 在全部训练集上重训 + 测试评估（携带样本权重）
# -------------------------------------------------------
final_model = clone(best_model)
final_model.fit(X_train, y_train_enc, sample_weight=sample_weight_train)

# 保存模型和标准化器
joblib.dump(final_model, os.path.join(results_dir, 'pam50_svm_model.pkl'))
joblib.dump(le,          os.path.join(results_dir, 'label_encoder.pkl'))
joblib.dump(scaler,      os.path.join(results_dir, 'standard_scaler.pkl'))
# 兼容名
joblib.dump(final_model, os.path.join(OUT_DIR, 'svm_best.pkl'))

# 测试集
y_pred_test = final_model.predict(X_test)
acc_test = accuracy_score(y_test_enc, y_pred_test)
f1w_test  = f1_score(y_test_enc, y_pred_test, average='weighted')
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
plt.title('Final Test Set Confusion Matrix (SVM)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'final_test_confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 兼容性旧图
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (SVM)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=220, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# 9. 分支信息（SVM无特征重要性）
# -------------------------------------------------------
# 保存分支信息（便于与多分支融合流水线兼容）
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
    'random_search_iters': RANDOM_SEARCH_ITERS
}
with open(os.path.join(results_dir, 'svm_branch_info.json'), 'w', encoding='utf-8') as f:
    json.dump(branch_info, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------
# 10. 结果文件汇总
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

# 训练过程指标（兼容下游）
training_metrics_df = pd.DataFrame([
    {'epoch': i, 'train_loss': 'N/A', 'train_acc': 'N/A', 'val_loss': 'N/A', 'val_acc': r['accuracy'], 'val_f1': r['f1_score']}
    for i, r in enumerate(fold_results, 1)
])
training_metrics_df.to_csv(os.path.join(results_dir, 'final_training_metrics.csv'), index=False)

print(f"\n=== 所有结果已保存 ===")
print(f"结果目录: {results_dir}")
print(f"包含文件:")
print(f"  - cv_results.csv / cv_summary.txt")
print(f"  - final_test_classification_report.csv / final_test_confusion_matrix.png")
print(f"  - cv_folds/  每折详细报告与混淆矩阵")
print(f"  - svm_branch_info.json")
print(f"  - pam50_svm_model.pkl / label_encoder.pkl / standard_scaler.pkl")
print(f"根目录兼容文件: svm_best.pkl / confusion_matrix.png")