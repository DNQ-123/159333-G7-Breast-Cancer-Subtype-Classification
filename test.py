# import pandas as pd
# import numpy as np
# import os
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.utils import shuffle
#
# # ----------------------
# # 1. 数据加载与格式转换
# # ----------------------
# def load_data(train_path, test_path):
#     """
#     加载训练集和测试集，转换为样本-基因矩阵格式
#     """
#     # 加载原始数据（基因作为行，样本作为列）
#     train_df = pd.read_csv(train_path, sep=',')  # 替换为实际文件路径
#     test_df = pd.read_csv(test_path, sep=',')    # 替换为实际文件路径
#
#     # 转置数据：样本作为行，基因作为列
#     train_df_t = train_df.T  # 转置训练集
#     test_df_t = test_df.T    # 转置测试集
#
#     # 提取列名（转置后第一行为基因名）
#     train_df_t.columns = train_df_t.iloc[0]  # 第一行作为列名（基因符号）
#     test_df_t.columns = test_df_t.iloc[0]
#
#     # 去除第一行（已作为列名），保留样本数据
#     train_df_t = train_df_t.iloc[1:]
#     test_df_t = test_df_t.iloc[1:]
#
#     # 提取标签（PAM50_Subtype列）
#     # 注意：原始数据中"PAM50_Subtype"所在行需单独提取
#     train_labels = train_df_t['PAM50_Subtype']  # 训练集标签
#     test_labels = test_df_t['PAM50_Subtype']    # 测试集标签
#
#     # 去除标签列，保留特征（基因表达值）
#     train_features = train_df_t.drop(columns=['PAM50_Subtype'])
#     test_features = test_df_t.drop(columns=['PAM50_Subtype'])
#
#     # 转换数据类型为数值型
#     train_features = train_features.astype(float)
#     test_features = test_features.astype(float)
#
#     return train_features, train_labels, test_features, test_labels
#
# # ----------------------
# # 2. 数据预处理
# # ----------------------
# def preprocess_data(train_features, test_features):
#     """
#     数据清洗、归一化和特征选择
#     """
#     # 处理缺失值（用列均值填充）
#     train_features = train_features.fillna(train_features.mean())
#     test_features = test_features.fillna(test_features.mean())
#
#     # 去除低方差基因（保留方差>0.1的特征）
#     selector = VarianceThreshold(threshold=0.1)
#     train_selected = selector.fit_transform(train_features)
#     test_selected = selector.transform(test_features)
#
#     # 提取筛选后的基因名
#     selected_genes = train_features.columns[selector.get_support()]
#
#     # 标准化（均值为0，方差为1）
#     scaler = StandardScaler()
#     train_scaled = scaler.fit_transform(train_selected)
#     test_scaled = scaler.transform(test_selected)
#
#     # 转换回DataFrame，保留基因名
#     train_processed = pd.DataFrame(train_scaled, columns=selected_genes, index=train_features.index)
#     test_processed = pd.DataFrame(test_scaled, columns=selected_genes, index=test_features.index)
#
#     return train_processed, test_processed, scaler, selector
#
# # ----------------------
# # 3. 模型训练与评估
# # ----------------------
# def train_and_evaluate(train_X, train_y, test_X, test_y):
#     """
#     训练多个分类模型并评估性能
#     """
#     # 定义模型字典
#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42),
#         "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
#         "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#         "KNN": KNeighborsClassifier(n_neighbors=5),
#         "SVM": SVC(kernel='linear', probability=True, random_state=42)
#     }
#
#     # 创建保存结果的目录
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("results", exist_ok=True)
#
#     # 训练并评估每个模型
#     best_model = None
#     best_accuracy = 0
#
#     for name, model in models.items():
#         print(f"----- 训练 {name} -----")
#         # 训练模型
#         model.fit(train_X, train_y)
#
#         # 预测
#         y_pred = model.predict(test_X)
#
#         # 评估指标
#         accuracy = accuracy_score(test_y, y_pred)
#         cm = confusion_matrix(test_y, y_pred)
#         report = classification_report(test_y, y_pred)
#
#         # 保存模型
#         joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.pkl")
#
#         # 保存评估结果
#         with open(f"results/{name}_report.txt", "w") as f:
#             f.write(f"Accuracy: {accuracy:.4f}\n")
#             f.write("Classification Report:\n")
#             f.write(report)
#
#         # 绘制混淆矩阵
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=model.classes_, yticklabels=model.classes_)
#         plt.xlabel("预测亚型")
#         plt.ylabel("实际亚型")
#         plt.title(f"{name} 混淆矩阵")
#         plt.savefig(f"results/{name}_confusion_matrix.png")
#         plt.close()
#
#         # 跟踪最佳模型
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = name
#
#         print(f"{name} 准确率: {accuracy:.4f}\n")
#
#     print(f"最佳模型: {best_model} (准确率: {best_accuracy:.4f})")
#     return models
#
# # ----------------------
# # 4. 主函数（执行流程）
# # ----------------------
# def main():
#     # 1. 加载数据（替换为您的文件路径）
#     train_path = "train_dataset_resolved_20250714_222423.csv"  # 训练集路径
#     test_path = "test_dataset_resolved_20250714_222423.csv"    # 测试集路径
#     train_features, train_labels, test_features, test_labels = load_data(train_path, test_path)
#
#     # 2. 数据预处理
#     train_processed, test_processed, scaler, selector = preprocess_data(train_features, test_features)
#     # 保存预处理工具
#     joblib.dump(scaler, "models/scaler.pkl")
#     joblib.dump(selector, "models/selector.pkl")
#
#     # 3. 训练与评估模型
#     models = train_and_evaluate(
#         train_processed, train_labels,
#         test_processed, test_labels
#     )
#
#     # 4. 输出样本分布
#     print("\n训练集亚型分布:")
#     print(train_labels.value_counts())
#     print("\n测试集亚型分布:")
#     print(test_labels.value_counts())
#
# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
"""
Breast-cancer PAM50 subtyping with Random Forest
使用独立训练集 & 测试集
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix)

import joblib  # 用于保存模型/结果

# 1. 路径
TRAIN_PATH = 'train_dataset_resolved_20250714_222423.csv'
TEST_PATH  = 'test_dataset_resolved_20250714_222423.csv'

# 2. 读取数据 ----------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH, index_col=0).T   # 行=样本, 列=基因(+标签)
test_df  = pd.read_csv(TEST_PATH,  index_col=0).T

# 2.1 提取标签
y_train = train_df['PAM50_Subtype']
y_test  = test_df['PAM50_Subtype']

# 2.2 对齐特征：以训练集基因为准，测试集缺失基因填 0
gene_cols = train_df.columns.drop('PAM50_Subtype')
X_train = train_df[gene_cols].astype(float)
X_test  = test_df.reindex(columns=gene_cols, fill_value=0).astype(float)

print('训练集形状:', X_train.shape, '类别分布:\n', y_train.value_counts())
print('测试集形状 :', X_test.shape,  '类别分布:\n', y_test.value_counts())

# 3. 建立 Pipeline -----------------------------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),   # 仅对训练集拟合
    ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1))
])

# 4. 超参数网格（可酌情扩大/缩小）
param_grid = {
    'clf__n_estimators': [200, 400, 600],
    'clf__max_depth':    [None, 30, 50],
    'clf__min_samples_split': [2, 5, 10],
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
print('\n分类报告:\n', classification_report(y_test, y_pred))
print('\n混淆矩阵:\n',
      pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=sorted(y_test.unique()),
                   columns=sorted(y_test.unique())))

# 7. 保存模型与 Top 特征 --------------------------------------------------
joblib.dump(best_model, 'pam50_rf_final.pkl')
print('\n模型已保存：pam50_rf_final.pkl')

# Top 20 重要基因
importances = best_model.named_steps['clf'].feature_importances_
top_genes = pd.Series(importances, index=gene_cols)\
             .sort_values(ascending=False).head(20)
print('\nTop 20 重要基因:\n', top_genes)
top_genes.to_csv('top20_genes.csv', header=['importance'])