# -*- coding: utf-8 -*-
"""
针对 Mammon2 数据集的 SVM 乳腺癌分子亚型分类
运行：
    python SVM.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, cohen_kappa_score)
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import uniform, loguniform

# ========= 1. 全局路径 =========
CSV_PATH = r'F:\massey\Mammon2\wsi_feature_labels.csv'
H5_ROOT  = r'F:\massey\Mammon2'
OUT_DIR  = r'./outputs_svm'
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

# ========= 3.1 特征标准化 =========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))

# ========= 4. 随机搜索 + SVM =========
svm_model = SVC(
    decision_function_shape='ovo',
    probability=True,          # 若想用 predict_proba 需置 True
    random_state=RANDOM_STATE
)

param_dist = {
    'C':      loguniform(1e-2, 1e3),
    'gamma':  loguniform(1e-4, 1e-1),
    'kernel': ['rbf', 'poly', 'sigmoid']
}

search = RandomizedSearchCV(
    svm_model, param_dist, n_iter=30, cv=5, scoring='f1_macro',
    n_jobs=-1, verbose=1, random_state=RANDOM_STATE
)

search.fit(X_train, y_train_enc)

best_model = search.best_estimator_
joblib.dump(best_model, os.path.join(OUT_DIR, 'svm_best.pkl'))
joblib.dump(le, os.path.join(OUT_DIR, 'label_encoder.pkl'))

# ========= 5. 评估 =========
y_pred = best_model.predict(X_test)
acc   = accuracy_score(y_test_enc, y_pred)
kappa = cohen_kappa_score(y_test_enc, y_pred)

# ---- 拼成待打印字符串 ----
report_str = classification_report(y_test_enc, y_pred,
                                   target_names=le.classes_, digits=4)
log_str = (f'Accuracy: {acc:.4f}  Kappa: {kappa:.4f}\n\n'
           f'Classification Report:\n{report_str}\n')

# ---- 既打印又保存 ----
print(log_str)
with open(os.path.join(OUT_DIR, 'metricsXGBoost.txt'), 'w', encoding='utf-8') as f:
    f.write(log_str)

# 混淆矩阵
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix(SVM)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=300)

print(f'\n所有结果已保存至 --> {OUT_DIR}')