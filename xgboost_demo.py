# -*- coding: utf-8 -*-
"""
Breast-cancer PAM50 subtyping with XGBoost
使用独立训练集 & 测试集（无警告版本）
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
import joblib  # 用于保存模型/结果

# 1. 路径
TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# 2. 读取数据 ----------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T   # 行=样本, 列=基因(+标签)
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# 2.1 标签编码：将字符串类型的类别转为整数编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['PAM50_Subtype'])
y_test  = label_encoder.transform(test_df['PAM50_Subtype'])
label_names = label_encoder.classes_

# 2.2 对齐特征：以训练集基因为准，测试集缺失基因填 0
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train = train_df[gene_cols].astype(float)
X_test  = test_df.reindex(columns=gene_cols, fill_value=0).astype(float)

print('训练集形状:', X_train.shape, '类别分布:\n', pd.Series(y_train).value_counts())
print('测试集形状 :', X_test.shape,  '类别分布:\n', pd.Series(y_test).value_counts())

# 3. 建立 Pipeline -----------------------------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(
        objective='multi:softprob',      # 多分类任务
        eval_metric='mlogloss',          # 必须设置以避免警告
        random_state=42,
        n_jobs=-1
    ))
])

# 4. 超参数网格
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6, 10],
    'clf__learning_rate': [0.01, 0.1, 0.3],
    'clf__subsample': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1)

# 5. 训练 ---------------------------------------------------------------
grid.fit(X_train, y_train)
print('\n最佳参数:', grid.best_params_)
print('交叉验证最佳准确率:', grid.best_score_)

# 6. 在独立测试集评估 ----------------------------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print('\n=== 独立测试集结果 ===')
print('准确率:', accuracy_score(y_test, y_pred))
print('\n分类报告:\n', classification_report(y_test, y_pred, target_names=label_names))
print('\n混淆矩阵:\n',
      pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=label_names,
                   columns=label_names))

# 7. 保存模型与 Top 特征 --------------------------------------------------
joblib.dump(best_model, 'pam50_xgb_final.pkl')
print('\n模型已保存：pam50_xgb_final.pkl')

# Top 20 重要基因
importances = best_model.named_steps['clf'].feature_importances_
top_genes = pd.Series(importances, index=gene_cols)\
             .sort_values(ascending=False).head(20)
print('\nTop 20 重要基因:\n', top_genes)
top_genes.to_csv('top20_genes_xgb.csv', header=['importance'])
