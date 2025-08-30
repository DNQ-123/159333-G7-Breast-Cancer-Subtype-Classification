# -*- coding: utf-8 -*-
"""
Fast mRMR + XGBoost for PAM50 subtyping
- 5-fold CV (修改为与三分支模型一致)
- 40 iterations RandomizedSearchCV
- 搜索结束把树数加到 1000 再在全训练集拟合一次
- 结果保存格式与三分支模型一致
"""
import warnings, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier         
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import randint, uniform
import joblib
import os
import json
from datetime import datetime

# --------------------
# 0. Reproducibility
# --------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
# 1. File paths
# --------------------
TRAIN_PATH = 'DATA/train_dataset_normalized_counts_2.csv'
TEST_PATH  = 'DATA/test_dataset_normalized_counts_2.csv'

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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

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

# --------------------
# 5.5. 详细的5折交叉验证评估
# --------------------
best_model = random_search.best_estimator_

# 创建结果保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_mrmr_xgboost_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "cv_folds"), exist_ok=True)

print(f"\n结果将保存到: {results_dir}")

# 详细的5折交叉验证
fold_results = []
cv_accuracies = []
cv_f1_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\n=== Fold {fold_idx}/5 ===")
    
    # 分割数据
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    # 训练模型（创建新实例避免冲突）
    fold_model = clone(best_model)
    fold_model.fit(X_fold_train, y_fold_train)
    
    # 预测
    y_fold_pred = fold_model.predict(X_fold_val)
    
    # 计算指标
    fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
    fold_f1 = f1_score(y_fold_val, y_fold_pred, average='weighted')
    
    cv_accuracies.append(fold_accuracy)
    cv_f1_scores.append(fold_f1)
    
    fold_results.append({
        'fold': fold_idx,
        'accuracy': fold_accuracy,
        'f1_score': fold_f1
    })
    
    print(f"Fold {fold_idx} - Accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f}")
    
    # 保存每折的分类报告
    fold_report = classification_report(y_fold_val, y_fold_pred, 
                                      target_names=label_names, 
                                      output_dict=True, zero_division=0)
    fold_report_df = pd.DataFrame(fold_report).transpose()
    fold_report_df.to_csv(os.path.join(results_dir, "cv_folds", f"fold_{fold_idx}_classification_report.csv"))
    
    # 保存每折的混淆矩阵
    fold_cm = confusion_matrix(y_fold_val, y_fold_pred)
    fold_cm_df = pd.DataFrame(fold_cm, index=label_names, columns=label_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(fold_cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Fold {fold_idx} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cv_folds", f"fold_{fold_idx}_confusion_matrix.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n=== 5-Fold CV Summary ===")
print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print(f"Mean F1-Score: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")

# --------------------
# 6. 把树数加到 1000，全训练集再拟合
# --------------------
best_model.set_params(clf__n_estimators=1000)
best_model.fit(X_train, y_train)

# --------------------
# 7. 最终测试集评估
# --------------------
y_pred = best_model.predict(X_test)

print('\n=== Final Test Results ===')
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test F1-Score: {test_f1:.4f}')

# 生成详细的分类报告
test_report = classification_report(y_test, y_pred, target_names=label_names, 
                                  output_dict=True, zero_division=0)
test_report_df = pd.DataFrame(test_report).transpose()

# 保存最终测试分类报告
test_report_df.to_csv(os.path.join(results_dir, "final_test_classification_report.csv"))
print('Detailed Report:\n', classification_report(y_test, y_pred, target_names=label_names))

# 生成和保存最终测试混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
print('\nConfusion matrix (count):\n', cm_df)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Final Test Set Confusion Matrix (mRMR + XGBoost)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'final_test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 同时保存旧格式的混淆矩阵（兼容性）
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (mRMR + XGBoost, fast)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_mrmr_xgb_fast.png', dpi=220, bbox_inches='tight')
plt.close()

# --------------------
# 8. 保存所有结果文件（标准格式）
# --------------------

# 8.1 CV结果文件
cv_results_df = pd.DataFrame(fold_results)
cv_results_df.to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

# 8.2 CV摘要文件
cv_summary_content = f"""5-Fold Cross-Validation Summary - mRMR + XGBoost
========================================
Architecture: mRMR Feature Selection + XGBoost Classifier
========================================
Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}
Mean F1-Score: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}
Individual Fold Results:"""

for i, result in enumerate(fold_results):
    cv_summary_content += f"\n  Fold {result['fold']}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}"

cv_summary_content += f"""

Final Test Performance:
  Test Accuracy: {test_accuracy:.4f}
  Test F1-Score: {test_f1:.4f}

Model Details:
  Best Parameters: {random_search.best_params_}
  CV Search Score: {random_search.best_score_:.4f}
"""

with open(os.path.join(results_dir, "cv_summary.txt"), 'w', encoding='utf-8') as f:
    f.write(cv_summary_content)

# 8.3 保存选择的基因
mrmr_step = best_model.named_steps['mrmr']
if hasattr(mrmr_step, 'selected_features_') and mrmr_step.selected_features_ is not None:
    selected_genes = pd.Index(mrmr_step.selected_features_)
else:
    support_idx = mrmr_step.get_support(indices=True)
    selected_genes = pd.Index(np.array(gene_cols)[support_idx])

# 保存选择的基因到结果目录
pd.Series(selected_genes, name='selected_gene').to_csv(os.path.join(results_dir, 'mrmr_selected_genes.csv'), index=False)
# 同时保存到根目录（兼容性）
pd.Series(selected_genes, name='selected_gene').to_csv('mrmr_selected_genes.csv', index=False)

# 8.4 保存XGBoost分支信息
xgb = best_model.named_steps['clf']
imp_series = pd.Series(xgb.feature_importances_, index=selected_genes).sort_values(ascending=False)

xgboost_info = {
    "model_type": "mRMR + XGBoost",
    "n_selected_features": len(selected_genes),
    "best_params": random_search.best_params_,
    "cv_search_score": float(random_search.best_score_),
    "final_test_accuracy": float(test_accuracy),
    "final_test_f1": float(test_f1),
    "top_10_features": {
        str(gene): float(importance) 
        for gene, importance in imp_series.head(10).items()
    },
    "feature_importance_stats": {
        "mean": float(imp_series.mean()),
        "std": float(imp_series.std()),
        "min": float(imp_series.min()),
        "max": float(imp_series.max())
    }
}

with open(os.path.join(results_dir, "xgboost_branch_info.json"), 'w', encoding='utf-8') as f:
    json.dump(xgboost_info, f, indent=2, ensure_ascii=False)

# 8.5 保存top-20特征重要性（兼容性）
imp_series.head(20).to_csv('top20_genes.csv', header=['importance'])

# 8.6 保存训练指标（模拟格式）
training_metrics = []
for i, result in enumerate(fold_results):
    training_metrics.append({
        'fold': result['fold'],
        'epoch': 'final',  # XGBoost没有epoch概念
        'train_loss': 'N/A',
        'train_acc': 'N/A', 
        'val_loss': 'N/A',
        'val_acc': result['accuracy'],
        'val_f1': result['f1_score']
    })

training_metrics_df = pd.DataFrame(training_metrics)
training_metrics_df.to_csv(os.path.join(results_dir, "final_training_metrics.csv"), index=False)

# 8.7 保存模型文件
joblib.dump(best_model, os.path.join(results_dir, 'pam50_xgb_mrmr_model.pkl'))
# 同时保存到根目录（兼容性）
joblib.dump(best_model, 'pam50_xgb_mrmr_fast.pkl')

# 8.8 打包复现文件
bundle_data = {
    "model": best_model, 
    "genes_order": list(gene_cols),
    "train_median": train_median, 
    "label_names": list(label_names),
    "selected_genes": list(selected_genes),
    "cv_results": fold_results,
    "test_results": {
        "accuracy": float(test_accuracy),
        "f1_score": float(test_f1)
    }
}

joblib.dump(bundle_data, os.path.join(results_dir, "xgb_mrmr_complete_bundle.pkl"))
# 同时保存到根目录（兼容性）
joblib.dump(
    {"model": best_model, "genes_order": list(gene_cols),
     "train_median": train_median, "label_names": list(label_names)},
    "xgb_mrmr_fast_bundle.pkl"
)

print(f"\n=== 所有结果已保存 ===")
print(f"结果目录: {results_dir}")
print(f"包含文件:")
print(f"  - cv_results.csv: 交叉验证结果")
print(f"  - cv_summary.txt: CV摘要统计")
print(f"  - final_test_classification_report.csv: 测试集分类报告")
print(f"  - final_test_confusion_matrix.png: 测试集混淆矩阵")
print(f"  - cv_folds/: 每折详细结果")
print(f"  - mrmr_selected_genes.csv: 选择的基因")
print(f"  - xgboost_branch_info.json: XGBoost详细信息")
print(f"  - final_training_metrics.csv: 训练指标")
print(f"  - pam50_xgb_mrmr_model.pkl: 训练好的模型")
print(f'\nSelected {len(selected_genes)} genes')
print(f'Root directory files: pam50_xgb_mrmr_fast.pkl, xgb_mrmr_fast_bundle.pkl, etc.')