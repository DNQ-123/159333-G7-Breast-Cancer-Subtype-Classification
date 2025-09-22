# -*- coding: utf-8 -*-
"""
针对 Mammon2 数据集的 randomforest 乳腺癌分子亚型分类
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
                             confusion_matrix, cohen_kappa_score)
from sklearn.ensemble import RandomForestClassifier   
import matplotlib
matplotlib.use('Agg')  # 无图形界面服务器也能画图
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
    """返回 mean-pool 后的 1-D 向量"""
    with h5py.File(path, 'r') as f:
        # 若您的 key 不是 'features' 请改这里
        data = f['features'][:]
    return data.mean(axis=0)   # (D,)

def build_Xy(csv_path, h5_root):
    df = pd.read_csv(csv_path)
    
    keep_classes = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Solid Tissue Normal']
    df = df[df['Label'].isin(keep_classes)].reset_index(drop=True)

    feats, labels, pids = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='读取+聚合'):
        rel_path = str(row['File_Path']).strip()
        rel_path = os.path.normpath(rel_path)
        h5_path  = os.path.abspath(os.path.join(h5_root, rel_path))

        if not os.path.isfile(h5_path):
            print(f'⚠️  跳过缺失文件: {h5_path}')
            continue

        vec = aggregate_h5(h5_path)
        feats.append(vec)
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

# ========= 4. 随机搜索 + RandomForest =========

rf_model = RandomForestClassifier(
    n_estimators=600,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    oob_score=False
)

param_dist = {
    'max_depth': [6, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.8]
}

search = RandomizedSearchCV(
    rf_model, param_dist, n_iter=30, cv=5, scoring='f1_macro',
    n_jobs=-1, verbose=1, random_state=RANDOM_STATE
)

# RandomForest 不需要 eval_set，把多余参数去掉
search.fit(X_train, y_train_enc)

best_model = search.best_estimator_
joblib.dump(best_model, os.path.join(OUT_DIR, 'rf_best.pkl'))
joblib.dump(le, os.path.join(OUT_DIR, 'label_encoder.pkl'))

# ========= 5. 评估 =========
y_pred = best_model.predict(X_test)
acc  = accuracy_score(y_test_enc, y_pred)
kappa = cohen_kappa_score(y_test_enc, y_pred)

print('\nAccuracy: {:.4f}  Kappa: {:.4f}'.format(acc, kappa))
print(classification_report(y_test_enc, y_pred,
                            target_names=le.classes_, digits=4))

# 混淆矩阵
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix(rf)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=300)

# 学习曲线
results = best_model.evals_result()
plt.figure()
plt.plot(results['validation_0']['mlogloss'], label='Test')
plt.title('Learning Curve (log-loss)')
plt.legend()
plt.savefig(os.path.join(OUT_DIR, 'learning_curve.png'), dpi=300)

print(f'\n所有结果已保存至 --> {OUT_DIR}')