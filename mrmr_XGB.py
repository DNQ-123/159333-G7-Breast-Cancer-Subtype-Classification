# -*- coding: utf-8 -*-
"""
Fast mRMR + XGBoost for PAM50 subtyping
- 3-fold CV
- 40 iterations RandomizedSearchCV
- 搜索结束把树数加到 1000 再在全训练集拟合一次
"""
import warnings, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier          # 关键替换
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import randint, uniform
import joblib

# --------------------
# 0. Reproducibility
# --------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
# 1. File paths
# --------------------
TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# --------------------
# 2. Load & basic preprocessing
# --------------------
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['PAM50_Subtype'])
y_test  = le.transform(test_df['PAM50_Subtype'])
label_names = le.classes_

# 对齐列顺序；把非数字转为 NaN；用“训练集每列中位数”填充 train/test 的缺失
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train_raw = train_df[gene_cols].apply(pd.to_numeric, errors='coerce')
X_test_raw  = test_df.reindex(columns=gene_cols).apply(pd.to_numeric, errors='coerce')
train_median = X_train_raw.median(axis=0)
X_train = X_train_raw.fillna(train_median)
X_test  = X_test_raw.fillna(train_median)

print('Train shape:', X_train.shape, '\nLabel counts:\n', pd.Series(y_train).value_counts())
print('Test  shape:', X_test.shape,  '\nLabel counts:\n', pd.Series(y_test).value_counts())

# -------------------------------------------------------
# 3. mRMR selector（与 RF 版本完全一致）
# -------------------------------------------------------
class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=100, redundancy_weight=1.0, random_state=None, discrete_target=True):
        self.k = k
        self.redundancy_weight = redundancy_weight
        self.random_state = random_state
        self.discrete_target = discrete_target

    @staticmethod
    def _to_numpy(X):
        return X.values if hasattr(X, "values") else np.asarray(X)

    def fit(self, X, y):
        k = int(self.k)
        redundancy_weight = float(self.redundancy_weight)
        X = self._to_numpy(X)
        n_samples, n_features = X.shape

        # relevance（互信息）
        self.mi_ = mutual_info_classif(
            X, y, random_state=self.random_state, discrete_features=False
        )

        # abs Pearson corr（冗余）
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        abs_corr = np.abs(corr)

        # greedy 选特征
        selected = [int(np.argmax(self.mi_))]
        while len(selected) < min(k, n_features):
            sel_idx = np.array(selected, dtype=int)
            redundancies = np.mean(abs_corr[:, sel_idx], axis=1)
            scores = self.mi_ - redundancy_weight * redundancies
            scores[sel_idx] = -np.inf
            j = int(np.argmax(scores))
            selected.append(j)

        self.selected_indices_ = np.array(selected, dtype=int)
        self.selected_features_ = None
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            cols = np.array(X.columns)
            self.selected_features_ = cols[self.selected_indices_].tolist()
            return X.iloc[:, self.selected_indices_]
        else:
            return self._to_numpy(X)[:, self.selected_indices_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices_
        mask = np.zeros_like(self.mi_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

# --------------------
# 4. Pipeline（XGBoost 部分）
# --------------------
pipe = Pipeline([
    ('mrmr', MRMRSelector(k=100, redundancy_weight=1.0, random_state=RANDOM_STATE)),
    ('clf', XGBClassifier(
        n_estimators=500,          # 搜索期先用较小树数
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

# --------------------
# 5. Fast RandomizedSearchCV（参数空间适配 XGBoost）
# --------------------
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_STATE)

param_dist = {
    # mRMR
    'mrmr__k': randint(60, 140),
    'mrmr__redundancy_weight': uniform(0.3, 1.0),

    # XGBoost
    'clf__n_estimators': randint(300, 900),
    'clf__max_depth': randint(3, 10),
    'clf__learning_rate': uniform(0.01, 0.2),
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__min_child_weight': randint(1, 6),
    'clf__gamma': uniform(0, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=40,
    scoring='balanced_accuracy',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("\n[Fast Search] RandomizedSearchCV running ...")
random_search.fit(X_train, y_train)
print("\nBest params:", random_search.best_params_)
print("Best CV balanced_acc:", random_search.best_score_)

# 可选：同一 CV 再评一遍
best_model = random_search.best_estimator_
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv,
                            scoring='balanced_accuracy', n_jobs=-1)
print("\nCV balanced_acc:", cv_scores)
print("Mean: {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))

# --------------------
# 6. 把树数加到 1000，全训练集再拟合
# --------------------
best_model.set_params(clf__n_estimators=1000)
best_model.fit(X_train, y_train)

# --------------------
# 7. Test evaluation
# --------------------
y_pred = best_model.predict(X_test)

print('\n=== Test Results ===')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nReport:\n', classification_report(y_test, y_pred, target_names=label_names))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
print('\nConfusion matrix (count):\n', cm_df)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (mRMR + XGBoost, fast)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_mrmr_xgb_fast.png', dpi=220, bbox_inches='tight')
plt.show()

# --------------------
# 8. Save model & outputs
# --------------------
joblib.dump(best_model, 'pam50_xgb_mrmr_fast.pkl')

# selected genes
mrmr_step = best_model.named_steps['mrmr']
if hasattr(mrmr_step, 'selected_features_') and mrmr_step.selected_features_ is not None:
    selected_genes = pd.Index(mrmr_step.selected_features_)
else:
    support_idx = mrmr_step.get_support(indices=True)
    selected_genes = pd.Index(np.array(gene_cols)[support_idx])
pd.Series(selected_genes, name='selected_gene').to_csv('mrmr_selected_genes.csv', index=False)
print(f'\nSelected {len(selected_genes)} genes -> mrmr_selected_genes.csv')

# top-20 importances
xgb = best_model.named_steps['clf']
imp_series = pd.Series(xgb.feature_importances_, index=selected_genes).sort_values(ascending=False)
imp_series.head(20).to_csv('top20_genes.csv', header=['importance'])
print('Top-20 importances -> top20_genes.csv')

# 打包复现
joblib.dump(
    {"model": best_model, "genes_order": list(gene_cols),
     "train_median": train_median, "label_names": list(label_names)},
    "xgb_mrmr_fast_bundle.pkl"
)
print("\nSaved: pam50_xgb_mrmr_fast.pkl & xgb_mrmr_fast_bundle.pkl")