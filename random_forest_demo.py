<<<<<<< HEAD
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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import joblib

# 1. File paths
TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# 2. Load & transpose (rows = samples, columns = genes + label)
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# 2.1 Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['PAM50_Subtype'])
y_test  = le.transform(test_df['PAM50_Subtype'])
label_names = le.classes_

# 2.2 Align features: use training genes, fill missing with 0
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train = train_df[gene_cols].astype(float)
X_test  = test_df.reindex(columns=gene_cols, fill_value=0).astype(float)

print('Train shape:', X_train.shape, '\nLabel counts:\n', pd.Series(y_train).value_counts())
print('Test shape :', X_test.shape,  '\nLabel counts:\n', pd.Series(y_test).value_counts())

# 3. Pipeline: scaling + random forest
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1))
])

# 4. Randomized hyperparameter search
param_dist = {
    'clf__n_estimators': randint(200, 1000),       # Number of trees
    'clf__max_depth': [10, 20, 30, 40, 50, None],  # Tree depth
    'clf__min_samples_split': randint(2, 10),      # Min samples to split
    'clf__min_samples_leaf': randint(1, 5),        # Min samples per leaf
    'clf__max_features': ['sqrt', 'log2']          # Feature selection strategy
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=50,               # Number of iterations
    scoring='accuracy',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# 5. Train model
random_search.fit(X_train, y_train)

# 6. Output best parameters and cross-validation score
print('\nBest parameters found by RandomizedSearchCV:\n', random_search.best_params_)
print('Best cross-validation accuracy:', random_search.best_score_)

# 7. Evaluate on independent test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print('\n=== Test Results ===')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nReport:\n', classification_report(y_test, y_pred, target_names=label_names))

# 8. Confusion matrix: DataFrame + heatmap
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

print('\nConfusion matrix (count):\n', cm_df)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 9. Save model and top features
joblib.dump(best_model, 'pam50_rf_final.pkl')

importances = best_model.named_steps['clf'].feature_importances_
top_genes = pd.Series(importances, index=gene_cols).sort_values(ascending=False).head(20)
print('\nTop 20 genes:\n', top_genes)
top_genes.to_csv('top20_genes.csv', header=['importance'])
=======
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
    'clf__n_estimators': [100, 300, 500, 800],
    'clf__max_depth': [10, 30, 50, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2']
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
>>>>>>> e299520c349a7bd7d0518372392bcbb8d42b395d
