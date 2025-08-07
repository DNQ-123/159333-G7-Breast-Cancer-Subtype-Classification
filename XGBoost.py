# -*- coding: utf-8 -*-
"""
Breast-cancer PAM50 subtyping with XGBoost
Independent training & testing sets
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# 1. read & transpose (rows = samples, cols = genes + label)
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# 2. encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['PAM50_Subtype'])
y_test  = label_encoder.transform(test_df['PAM50_Subtype'])
label_names = label_encoder.classes_

# 3. align features
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train = train_df[gene_cols].astype(float)
X_test  = test_df.reindex(columns=gene_cols, fill_value=0).astype(float)

print('Train shape:', X_train.shape, 'Label counts:\n', pd.Series(y_train).value_counts())
print('Test shape :', X_test.shape,  'Label counts:\n', pd.Series(y_test).value_counts())

# 4. pipeline: scaling + XGBoost
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(
        tree_method='hist',
        device='cuda',
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1))
])

param_grid = {
    'clf__n_estimators': [150, 300],
    'clf__max_depth': [None, 6],
    'clf__learning_rate': [0.1],
    'clf__subsample': [0.8]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=1,
    verbose=1)

grid.fit(X_train, y_train)
print('\nBest params:', grid.best_params_)
print('CV best accuracy:', grid.best_score_)

# 5. evaluate on independent test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('\n=== Test Results ===')
print('Accuracy:', acc)
print('\nReport:\n', classification_report(y_test, y_pred, target_names=label_names))

# 6. 绘制并保存混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names)

plt.title('PAM50 Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

# 先保存再显示，避免 show 阻塞导致文件没写完
plt.savefig('confusion_matrix_xgb.png', dpi=300, bbox_inches='tight')
plt.show()          # 默认 block=False

# 7. save
joblib.dump(best_model, 'pam50_xgb_final.pkl')
importances = best_model.named_steps['clf'].feature_importances_
top_genes = pd.Series(importances, index=gene_cols).sort_values(ascending=False).head(20)
print('\nTop 20 genes:\n', top_genes)
top_genes.to_csv('top20_genes_xgb.csv', header=['importance'])
