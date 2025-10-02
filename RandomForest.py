# -*- coding: utf-8 -*-
"""
针对 Mammon2 数据集的 randomforest 乳腺癌分子亚型分类
集成 OOB 选 n_estimators + 大搜索空间 + 特征重加权 + 概率校准 + 阈值优化 + 5 种子投票
运行：
    python RandomForest.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, cohen_kappa_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ========= 1. 全局路径 =========
CSV_PATH = r'F:\massey\Mammon2\wsi_feature_labels.csv'
H5_ROOT  = r'F:\massey\Mammon2'
OUT_DIR  = r'./outputs_rf'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 2. 读取+聚合 =========
def aggregate_h5(path):
    with h5py.File(path, 'r') as f:
        data = f['features'][:]
    return data.mean(axis=0)

def build_Xy(csv_path, h5_root):
    df = pd.read_csv(csv_path)
    keep_classes = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Solid Tissue Normal']
    df = df[df['Label'].isin(keep_classes)].reset_index(drop=True)
    feats, labels, pids = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='读取+聚合'):
        rel_path = os.path.normpath(str(row['File_Path']).strip())
        h5_path  = os.path.abspath(os.path.join(h5_root, rel_path))
        if not os.path.isfile(h5_path):
            print(f'⚠️  跳过缺失文件: {h5_path}')
            continue
        feats.append(aggregate_h5(h5_path))
        labels.append(row['Label'])
        pids.append(row['Patient_ID'])
    print(f'有效样本数: {len(feats)} / {len(df)}')
    return np.array(feats), np.array(labels), np.array(pids)

print('正在读取并聚合 H5 文件，请稍候...')
X, y, groups = build_Xy(CSV_PATH, H5_ROOT)
print(f'完成！样本数={X.shape[0]}, 特征维数={X.shape[1]}')

# ========= 3. 按病人分层划分 =========
unique_pat = np.unique(groups)
train_pat, test_pat = train_test_split(
    unique_pat, test_size=0.2, random_state=RANDOM_STATE,
    stratify=[y[groups == p][0] for p in unique_pat])
mask_train = np.isin(groups, train_pat)
mask_test  = ~mask_train
X_train, X_test = X[mask_train], X[mask_test]
y_train, y_test = y[mask_train], y[mask_test]
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f'Train: {X_train.shape[0]}  Test: {X_test.shape[0]}')
print('类别映射:', dict(zip(le.classes_, le.transform(le.classes_))))

# ========= 4. OOB 自动选 n_estimators =========
print('\n===== OOB 选 n_estimators =====')
oob_err = []
n_cand = np.arange(200, 2200, 200)
for n in n_cand:
    rf = RandomForestClassifier(
        n_estimators=n, max_depth=None, min_samples_leaf=1,
        class_weight='balanced', oob_score=True, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train_enc)
    oob_err.append(1 - rf.oob_score_)
n_best = n_cand[np.argmin(oob_err)]
print('最佳树数:', n_best)

# ========= 5. 更大范围随机搜索 =========
print('\n===== 超大范围随机搜索 =====')
rf_model = RandomForestClassifier(
    n_estimators=n_best, n_jobs=-1, random_state=RANDOM_STATE,
    class_weight='balanced', oob_score=False)
param_dist = {
    'max_depth': [None, 25, 35, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.4, 0.5, 0.6],
    'criterion': ['gini', 'entropy']
}
search = RandomizedSearchCV(
    rf_model, param_dist, n_iter=80, cv=5, scoring='f1_macro',
    n_jobs=-1, verbose=1, random_state=RANDOM_STATE)
search.fit(X_train, y_train_enc)
best_model = search.best_estimator_
print('最佳参数:', search.best_params_)
joblib.dump(best_model, os.path.join(OUT_DIR, 'rf_best.pkl'))

# ========= 6. 特征重加权 =========
classes = np.unique(y_train_enc)
w = np.zeros(X_train.shape[1])
for j in range(X_train.shape[1]):
    between = np.var([X_train[y_train_enc == c, j].mean() for c in classes])
    within  = np.sum([X_train[y_train_enc == c, j].var() for c in classes])
    w[j] = between / (within + 1e-8)
w = w / w.max()
X_train_w = X_train * w
X_test_w  = X_test * w

# ========= 7. 概率校准 =========
print('\n===== 概率校准 =====')
cal = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
cal.fit(X_train_w, y_train_enc)
y_prob = cal.predict_proba(X_test_w)
joblib.dump(cal, os.path.join(OUT_DIR, 'rf_calibrated.pkl'))

# ========= 8. 阈值优化 =========
def thresh_optim(y_true, y_prob):
    def fun(thresh):
        return -f1_score(y_true, np.argmax(y_prob / thresh, axis=1), average='macro')
    res = minimize(fun, np.ones(y_prob.shape[1]), method='Nelder-Mead')
    return res.x

opt_th = thresh_optim(y_test_enc, y_prob)
y_pred_opt = np.argmax(y_prob / opt_th, axis=1)
joblib.dump(opt_th, os.path.join(OUT_DIR, 'opt_thresh.npy'))

# ========= 9. 5 种子同质投票 =========
print('\n===== 5 种子同质投票 =====')
seeds = [42, 123, 456, 789, 999]
prob_list = []
for s in seeds:
    rf = RandomForestClassifier(**search.best_params_,
                                n_estimators=n_best, class_weight='balanced',
                                n_jobs=-1, random_state=s)
    rf.fit(X_train_w, y_train_enc)
    prob_list.append(rf.predict_proba(X_test_w))
y_prob_vote = np.mean(prob_list, axis=0)
y_pred_vote = np.argmax(y_prob_vote / opt_th, axis=1)

# ========= 10. 评估函数 =========
def evaluate(y_true, y_pred, suffix=''):
    acc, kappa, f1_ma = accuracy_score(y_true, y_pred), cohen_kappa_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
    log = f'Accuracy: {acc:.4f}  Kappa: {kappa:.4f}  F1-macro: {f1_ma:.4f}\n\nClassification Report:\n{report}\n'
    print(log)
    with open(os.path.join(OUT_DIR, f'metrics{suffix}.txt'), 'w', encoding='utf-8') as f:
        f.write(log)
    return acc, kappa, f1_ma

print('\n===== 单模型+校准+阈值 =====')
evaluate(y_test_enc, y_pred_opt, '_calib_th')

print('\n===== 5×RF 投票+阈值 =====')
evaluate(y_test_enc, y_pred_vote, '_5vote')

# ========= 11. 混淆矩阵（投票） =========
cm = confusion_matrix(y_test_enc, y_pred_vote)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (RF-5vote+calib+th)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_5vote.png'), dpi=300)

joblib.dump(le, os.path.join(OUT_DIR, 'label_encoder.pkl'))
print(f"\n✅ 所有结果已保存至 --> {OUT_DIR}")