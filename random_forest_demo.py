# -*- coding: utf-8 -*-
"""
Breast-cancer PAM50 subtyping with Random Forest
Independent training & testing sets
Enhanced with comprehensive result saving
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from scipy.stats import randint
import joblib

# 1. File paths - 修改为新的数据集
TRAIN_PATH = 'train_dataset_normalized_counts_2.csv'
TEST_PATH  = 'test_dataset_normalized_counts_2.csv'

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

# 创建结果保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_rf_pam50_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "cv_folds"), exist_ok=True)

print(f"\n结果将保存到: {results_dir}")

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
print("\n[RandomizedSearchCV] Running hyperparameter optimization...")
random_search.fit(X_train, y_train)

# 6. Output best parameters and cross-validation score
print('\nBest parameters found by RandomizedSearchCV:\n', random_search.best_params_)
print('Best cross-validation accuracy:', random_search.best_score_)

# 6.1 详细的5折交叉验证评估
best_model = random_search.best_estimator_
fold_results = []
cv_accuracies = []
cv_f1_scores = []

print("\n=== Detailed 5-Fold Cross-Validation ===")
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold_idx}/5 ---")
    
    # 分割数据
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    # 训练模型
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

# 7. 最终模型训练
print("\n=== Training Final Model ===")
final_model = clone(best_model)
final_model.fit(X_train, y_train)

# 8. Evaluate on independent test set
y_pred = final_model.predict(X_test)

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
plt.title('Final Test Set Confusion Matrix (Random Forest)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'final_test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9. 保存所有结果文件
# 9.1 CV结果文件
cv_results_df = pd.DataFrame(fold_results)
cv_results_df.to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

# 9.2 CV摘要文件
cv_summary_content = f"""5-Fold Cross-Validation Summary - Random Forest
========================================
Architecture: StandardScaler + Random Forest Classifier
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

# 9.3 保存特征重要性
importances = final_model.named_steps['clf'].feature_importances_
feature_importance_df = pd.DataFrame({
    'gene': gene_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

# 保存完整特征重要性
feature_importance_df.to_csv(os.path.join(results_dir, 'feature_importances.csv'), index=False)

# 保存top20基因（结果目录和根目录）
top_genes = feature_importance_df.head(20)
top_genes.to_csv(os.path.join(results_dir, 'top20_genes.csv'), index=False)
top_genes.to_csv('top20_genes.csv', index=False)  # 兼容性保存

# 9.4 保存随机森林分支信息
rf_info = {
    "model_type": "StandardScaler + RandomForest",
    "n_features": len(gene_cols),
    "best_params": random_search.best_params_,
    "cv_search_score": float(random_search.best_score_),
    "final_test_accuracy": float(test_accuracy),
    "final_test_f1": float(test_f1),
    "top_10_features": {
        row['gene']: float(row['importance']) 
        for _, row in feature_importance_df.head(10).iterrows()
    },
    "feature_importance_stats": {
        "mean": float(feature_importance_df['importance'].mean()),
        "std": float(feature_importance_df['importance'].std()),
        "min": float(feature_importance_df['importance'].min()),
        "max": float(feature_importance_df['importance'].max())
    }
}

with open(os.path.join(results_dir, "random_forest_info.json"), 'w', encoding='utf-8') as f:
    json.dump(rf_info, f, indent=2, ensure_ascii=False)

# 9.5 保存训练指标
training_metrics = []
for i, result in enumerate(fold_results):
    training_metrics.append({
        'fold': result['fold'],
        'accuracy': result['accuracy'],
        'f1_score': result['f1_score']
    })

training_metrics_df = pd.DataFrame(training_metrics)
training_metrics_df.to_csv(os.path.join(results_dir, "training_metrics.csv"), index=False)

# 9.6 保存模型文件
joblib.dump(final_model, os.path.join(results_dir, 'pam50_rf_final_model.pkl'))
joblib.dump(final_model, 'pam50_rf_final.pkl')  # 兼容性保存

# 9.7 打包复现文件
bundle_data = {
    "model": final_model, 
    "genes_order": list(gene_cols),
    "label_names": list(label_names),
    "cv_results": fold_results,
    "test_results": {
        "accuracy": float(test_accuracy),
        "f1_score": float(test_f1)
    }
}

joblib.dump(bundle_data, os.path.join(results_dir, "rf_complete_bundle.pkl"))
joblib.dump(bundle_data, "rf_final_bundle.pkl")  # 兼容性保存

print(f"\n=== 所有结果已保存 ===")
print(f"结果目录: {results_dir}")
print(f"包含文件:")
print(f"  - cv_results.csv: 交叉验证结果")
print(f"  - cv_summary.txt: CV摘要统计")
print(f"  - final_test_classification_report.csv: 测试集分类报告")
print(f"  - final_test_confusion_matrix.png: 测试集混淆矩阵")
print(f"  - cv_folds/: 每折详细结果（分类报告+混淆矩阵）")
print(f"  - feature_importances.csv: 特征重要性")
print(f"  - top20_genes.csv: Top 20基因")
print(f"  - random_forest_info.json: 模型详细信息")
print(f"  - training_metrics.csv: 训练指标")
print(f"  - pam50_rf_final_model.pkl: 训练好的模型")
print(f"  - rf_complete_bundle.pkl: 完整复现包")
print(f'\nRoot directory files: pam50_rf_final.pkl, rf_final_bundle.pkl, top20_genes.csv')

# 10. 显示最终混淆矩阵（可选）
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Random Forest)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_rf_final.png', dpi=220, bbox_inches='tight')
plt.show()