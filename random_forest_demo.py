# -*- coding: utf-8 -*-
"""
Breast-cancer PAM50 subtyping with Random Forest
Independent training & testing sets
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. file paths
TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# 2. load & transpose (rows=samples, cols=genes+label)
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# 2.1 encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['PAM50_Subtype'])
y_test  = le.transform(test_df['PAM50_Subtype'])
label_names = le.classes_

# 2.2 align features: use training genes, fill missing with 0
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train = train_df[gene_cols].astype(float)
X_test  = test_df.reindex(columns=gene_cols, fill_value=0).astype(float)

print('Train shape:', X_train.shape, '\nLabel counts:\n', pd.Series(y_train).value_counts())
print('Test shape :', X_test.shape,  '\nLabel counts:\n', pd.Series(y_test).value_counts())

# 3. pipeline: scaling + random forest
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1))
])

# 4. hyper-parameter grid
param_grid = {
    'clf__n_estimators': [400, 600, 800],
    'clf__max_depth':    [30, 50, 70],
    'clf__min_samples_split': [2, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1)

pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.max_rows', None)     # display all rows

# 5. train
grid.fit(X_train, y_train)
print('\nBest params:', grid.best_params_)
print('CV best accuracy:', grid.best_score_)

# 6. evaluate on independent test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print('\n=== Test Results ===')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nReport:\n', classification_report(y_test, y_pred, target_names=label_names))

# 7. enhanced confusion matrix: DataFrame + heatmap
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

print('\nConfusion matrix (count):\n', cm_df)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 8. save model & top features
joblib.dump(best_model, 'pam50_rf_final.pkl')

importances = best_model.named_steps['clf'].feature_importances_
top_genes = pd.Series(importances, index=gene_cols).sort_values(ascending=False).head(20)
print('\nTop 20 genes:\n', top_genes)
top_genes.to_csv('top20_genes.csv', header=['importance'])