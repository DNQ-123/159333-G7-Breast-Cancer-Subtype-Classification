# -*- coding: utf-8 -*-
"""
Breast-cancer PAM50 subtyping with XGBoost GPU (native interface)
Independent training & testing sets
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# 1. 路径
TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# 2. 读取并转置：行=样本，列=基因(+标签)
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# 3. 标签编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['PAM50_Subtype'])
y_test  = label_encoder.transform(test_df['PAM50_Subtype'])
label_names = label_encoder.classes_

# 4. 特征对齐
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train = train_df[gene_cols].astype(float)
X_test  = test_df.reindex(columns=gene_cols, fill_value=0).astype(float)

print('Train shape:', X_train.shape, 'Label counts:\n',
      pd.Series(y_train).value_counts())
print('Test shape :', X_test.shape,  'Label counts:\n',
      pd.Series(y_test).value_counts())

# 5. 构造 GPU 上的 DMatrix
dtrain = xgb.DMatrix(X_train.values, label=y_train)
dtest  = xgb.DMatrix(X_test.values,  label=y_test)

# 6. 训练参数
params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_train)),
    'tree_method': 'gpu_hist',   
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'seed': 42
}

# 7. 训练（带早停）
model = xgb.train(
    params,
    dtrain,
    num_boost_round=800,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=50
)

# 8. 预测
y_pred_prob = model.predict(dtest)
y_pred = np.argmax(y_pred_prob, axis=1)

# 9. 评估
acc = accuracy_score(y_test, y_pred)
print('\n=== Test Results (GPU) ===')
print('Accuracy:', acc)
print('\nReport:\n', classification_report(y_test, y_pred, target_names=label_names))

# 10. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.title('PAM50 Confusion Matrix (GPU)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_xgb_gpu.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 保存模型 & 特征重要性
model.save_model('pam50_xgb_gpu.json')        # 原生接口模型
importance = model.get_score(importance_type='gain')
top_genes = pd.Series(importance).sort_values(ascending=False).head(20)
print('\nTop 20 genes:\n', top_genes)
top_genes.to_csv('top20_genes_xgb_gpu.csv', header=['importance'])