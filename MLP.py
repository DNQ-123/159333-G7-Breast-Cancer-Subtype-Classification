#!/usr/bin/env python3
"""
MLP分类器 - 基于CLAM特征数据集
使用MLP对聚合后的特征进行分类，与MIL.py保持数据处理一致性
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from collections import Counter
import warnings
import torch.nn.functional as F
import gc
from tqdm import tqdm
from datetime import datetime
import json
import math

warnings.filterwarnings('ignore')

class MLPWSIDataset(Dataset):
    """
    MLP WSI数据集 - 聚合patch-level特征为WSI-level特征
    与MIL.py保持一致的数据加载方式，但对特征进行聚合
    """
    
    def __init__(self, csv_file, feature_dir, sample_types=['01', '11'], 
                 max_patches=2000, enable_augmentation=False, aggregation_method='mean'):
        """
        Args:
            csv_file: CSV文件路径
            feature_dir: 特征文件目录
            sample_types: 样本类型
            max_patches: 最大patch数量，用于内存控制
            enable_augmentation: 是否启用数据增强
            aggregation_method: 特征聚合方法 ('mean', 'max', 'mean_max')
        """
        self.data_df = pd.read_csv(csv_file, dtype={'Sample_Type_Code': str})
        self.feature_dir = Path(feature_dir)
        self.max_patches = max_patches
        self.enable_augmentation = enable_augmentation
        self.aggregation_method = aggregation_method
        self.training = False
        
        # 数据过滤 - 与MIL.py保持一致
        self.data_df = self.data_df[self.data_df['Sample_Type_Code'].isin(sample_types)]
        self.data_df = self.data_df[~self.data_df['Label'].isin(['Unknown', 'Metastatic'])]
        self.data_df = self.data_df.reset_index(drop=True)
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.data_df['encoded_label'] = self.label_encoder.fit_transform(self.data_df['Label'])
        
        # 检查数据完整性
        self._check_data_integrity()
        
        # 预计算特征维度
        self.base_feature_dim = self._get_feature_dimension()
        self.feature_dim = self._get_aggregated_feature_dim()
        
        print(f"MLP数据集初始化完成:")
        print(f"  样本数量: {len(self.data_df)}")
        print(f"  基础特征维度: {self.base_feature_dim}")
        print(f"  聚合后特征维度: {self.feature_dim}")
        print(f"  聚合方法: {self.aggregation_method}")
        print(f"  最大patch数: {self.max_patches}")
        self._print_label_distribution()
    
    def _check_data_integrity(self):
        """检查数据完整性"""
        missing_files = []
        print("检查数据文件完整性...")
        
        for idx in tqdm(range(min(len(self.data_df), 10)), desc="抽样检查"):
            filename = self.data_df.iloc[idx]['Filename']
            filepath = self.feature_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"警告: 发现 {len(missing_files)} 个缺失文件")
        else:
            print("✅ 数据文件完整性检查通过")
    
    def _get_feature_dimension(self):
        """获取基础特征维度"""
        for idx in range(min(5, len(self.data_df))):
            try:
                filename = self.data_df.iloc[idx]['Filename']
                filepath = self.feature_dir / filename
                
                with h5py.File(filepath, 'r') as f:
                    if 'features' in f.keys():
                        features = f['features'][:]
                    elif 'feats' in f.keys():
                        features = f['feats'][:]
                    else:
                        key = list(f.keys())[0]
                        features = f[key][:]
                
                # 处理特征形状
                if features.ndim == 3:
                    features = features.squeeze(0)
                
                if features.ndim == 2:
                    return features.shape[1]  # 返回特征维度
                else:
                    return len(features)
                    
            except Exception as e:
                print(f"跳过文件 {filepath}: {e}")
                continue
        
        return 1024  # 默认CLAM特征维度
    
    def _get_aggregated_feature_dim(self):
        """获取聚合后的特征维度"""
        if self.aggregation_method == 'mean':
            return self.base_feature_dim
        elif self.aggregation_method == 'max':
            return self.base_feature_dim
        elif self.aggregation_method == 'mean_max':
            return self.base_feature_dim * 2  # mean和max拼接
        else:
            return self.base_feature_dim
    
    def _print_label_distribution(self):
        """打印标签分布"""
        print("标签分布:")
        for label_name in self.label_encoder.classes_:
            count = np.sum(self.data_df['encoded_label'] == 
                          self.label_encoder.transform([label_name])[0])
            percentage = count / len(self.data_df) * 100
            print(f"  {label_name}: {count} files ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.data_df)
    
    def _aggregate_features(self, features):
        """聚合patch特征为WSI特征"""
        if self.aggregation_method == 'mean':
            return np.mean(features, axis=0)
        elif self.aggregation_method == 'max':
            return np.max(features, axis=0)
        elif self.aggregation_method == 'mean_max':
            mean_features = np.mean(features, axis=0)
            max_features = np.max(features, axis=0)
            return np.concatenate([mean_features, max_features])
        else:
            # 默认使用mean
            return np.mean(features, axis=0)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        filename = row['Filename']
        filepath = self.feature_dir / filename
        
        try:
            with h5py.File(filepath, 'r') as f:
                # 与MIL.py保持一致的特征加载方式
                if 'features' in f.keys():
                    features = f['features'][:]
                elif 'feats' in f.keys():
                    features = f['feats'][:]
                else:
                    key = list(f.keys())[0]
                    features = f[key][:]
            
            # 特征处理
            if features.ndim == 3:
                features = features.squeeze(0)
            
            # 确保是2D特征
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 限制patch数量以控制内存
            if features.shape[0] > self.max_patches:
                if self.training and self.enable_augmentation:
                    # 训练时随机采样
                    indices = np.random.choice(features.shape[0], self.max_patches, replace=False)
                    features = features[indices]
                else:
                    # 验证/测试时选择前N个
                    features = features[:self.max_patches]
            
            # 数据增强（仅在训练时）
            if self.enable_augmentation and self.training:
                features = self._augment_patches(features)
            
            # 聚合特征
            aggregated_features = self._aggregate_features(features)
            
            # 转换为tensor
            features_tensor = torch.FloatTensor(aggregated_features)
            label = torch.LongTensor([row['encoded_label']])[0]
            
            return features_tensor, label, filename
            
        except Exception as e:
            print(f"加载文件错误 {filepath}: {e}")
            # 返回默认特征
            zero_features = torch.zeros(self.feature_dim)
            label = torch.LongTensor([row['encoded_label']])[0]
            return zero_features, label, filename
    
    def _augment_patches(self, features):
        """patch级别的数据增强"""
        if np.random.rand() < 0.3:
            # 添加少量噪声
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise
        
        if np.random.rand() < 0.2:
            # patch级别的dropout
            num_patches = features.shape[0]
            keep_ratio = 0.9
            keep_patches = int(num_patches * keep_ratio)
            if keep_patches > 0:
                indices = np.random.choice(num_patches, keep_patches, replace=False)
                features = features[indices]
        
        return features
    
    def set_training_mode(self, training):
        """设置训练模式"""
        self.training = training

# ===============================
# MLP分类器模型
# ===============================

class MLPClassifierModel(nn.Module):
    """
    MLP分类器模型
    对聚合后的WSI特征进行分类
    """
    
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128], num_classes=5, dropout=0.25):
        super(MLPClassifierModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 聚合后的特征
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.mlp(x)

# ===============================
# MLP分类器主类
# ===============================

class MLPClassifier:
    """
    MLP分类器主类
    使用MLP对聚合后的WSI特征进行分类
    """
    
    def __init__(self, csv_file, feature_dir, aggregation_method='mean', device=None):
        self.csv_file = csv_file
        self.feature_dir = feature_dir
        self.aggregation_method = aggregation_method
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"特征聚合方法: {aggregation_method}")
        
        # 创建数据集
        self.dataset = MLPWSIDataset(
            csv_file, feature_dir, 
            max_patches=1000,  # 控制内存使用
            enable_augmentation=True,
            aggregation_method=aggregation_method
        )
        
        self.num_classes = len(self.dataset.label_encoder.classes_)
        self.class_names = self.dataset.label_encoder.classes_
        self.feature_dim = self.dataset.feature_dim
        
        # 创建结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results_mlp_{aggregation_method}_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        print(f"结果将保存到: {self.results_dir}")
        
        # 保存实验配置
        self.save_experiment_config()
    
    def save_experiment_config(self):
        """保存实验配置信息"""
        config = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': f'MLP_Classifier_{self.aggregation_method}',
            'aggregation_method': self.aggregation_method,
            'device': str(self.device),
            'num_classes': self.num_classes,
            'class_names': list(self.class_names),
            'feature_dim': self.feature_dim,
            'base_feature_dim': self.dataset.base_feature_dim,
            'dataset_info': {
                'total_samples': len(self.dataset),
                'max_patches': self.dataset.max_patches,
                'csv_file': str(self.csv_file),
                'feature_dir': str(self.feature_dir)
            },
            'class_distribution': {}
        }
        
        # 添加类别分布信息
        for label_name in self.class_names:
            count = np.sum(self.dataset.data_df['encoded_label'] == 
                          self.dataset.label_encoder.transform([label_name])[0])
            config['class_distribution'][label_name] = {
                'count': int(count),
                'percentage': float(count / len(self.dataset) * 100)
            }
        
        # 保存配置
        config_path = self.results_dir / 'experiment_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"实验配置已保存: {config_path}")
    
    def create_model(self, hidden_dims=[512, 256, 128]):
        """创建MLP模型"""
        model = MLPClassifierModel(
            input_dim=self.feature_dim,
            hidden_dims=hidden_dims,
            num_classes=self.num_classes,
            dropout=0.25
        )
        return model.to(self.device)
    
    def save_fold_results(self, fold, y_true, y_pred, fold_name="validation", y_pred_proba=None):
        """保存每折的结果"""
        fold_dir = self.results_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # 分类报告
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 保存分类报告
        report_path = fold_dir / f"fold_{fold+1}_{fold_name}_classification_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(class_report, f, indent=2, ensure_ascii=False)
        
        report_df = pd.DataFrame(class_report).transpose()
        report_csv_path = fold_dir / f"fold_{fold+1}_{fold_name}_classification_report.csv"
        report_df.to_csv(report_csv_path, index=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_path = fold_dir / f"fold_{fold+1}_{fold_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Fold {fold+1} ({fold_name.title()}) - MLP ({self.aggregation_method.upper()})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_plot_path = fold_dir / f"fold_{fold+1}_{fold_name}_confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算详细指标
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        metrics = {
            'fold': fold + 1,
            'fold_type': fold_name,
            'aggregation_method': self.aggregation_method,
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'total_samples': len(y_true),
            'correct_predictions': int(np.sum(y_true == y_pred))
        }
        
        metrics_path = fold_dir / f"fold_{fold+1}_{fold_name}_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 计算和绘制AUROC/AUPRC（如果提供了预测概率）
        if y_pred_proba is not None:
            self._plot_roc_prc_curves(fold, y_true, y_pred_proba, fold_name, fold_dir, metrics)
        
        return class_report, cm, metrics
    
    def _plot_roc_prc_curves(self, fold, y_true, y_pred_proba, fold_name, fold_dir, metrics):
        """绘制ROC和PRC曲线"""
        try:
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba)
            
            # 对于多分类问题，需要进行二值化
            n_classes = len(self.class_names)
            
            if n_classes == 2:
                # 二分类情况
                self._plot_binary_roc_prc(fold, y_true, y_pred_proba, fold_name, fold_dir, metrics)
            else:
                # 多分类情况
                self._plot_multiclass_roc_prc(fold, y_true, y_pred_proba, fold_name, fold_dir, metrics)
                
        except Exception as e:
            print(f"绘制ROC/PRC曲线时出错: {e}")
    
    def _plot_binary_roc_prc(self, fold, y_true, y_pred_proba, fold_name, fold_dir, metrics):
        """绘制二分类的ROC和PRC曲线"""
        # 使用正类的概率
        y_scores = y_pred_proba[:, 1]
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # PRC曲线
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # 绘制ROC曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold+1} ({fold_name.title()}) - MLP ({self.aggregation_method.upper()})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 绘制PRC曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Fold {fold+1} ({fold_name.title()}) - MLP ({self.aggregation_method.upper()})')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        roc_prc_path = fold_dir / f"fold_{fold+1}_{fold_name}_roc_prc_curves.png"
        plt.savefig(roc_prc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 更新metrics
        metrics['roc_auc'] = float(roc_auc)
        metrics['average_precision'] = float(avg_precision)
        
        print(f"  ROC AUC: {roc_auc:.4f}, Average Precision: {avg_precision:.4f}")
    
    def _plot_multiclass_roc_prc(self, fold, y_true, y_pred_proba, fold_name, fold_dir, metrics):
        """绘制多分类的ROC和PRC曲线"""
        n_classes = len(self.class_names)
        
        # 二值化标签
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # 计算每个类别的ROC和PRC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        
        # 计算macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # 计算macro-average precision
        avg_precision["macro"] = np.mean([avg_precision[i] for i in range(n_classes)])
        
        # 绘制ROC曲线
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        # 绘制每个类别的ROC曲线
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # 绘制macro-average ROC曲线
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=3,
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-class ROC Curves - Fold {fold+1} ({fold_name.title()}) - MLP ({self.aggregation_method.upper()})')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True, alpha=0.3)
        
        # 绘制PRC曲线
        plt.subplot(1, 2, 2)
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AP = {avg_precision[i]:.3f})')
        
        # 绘制macro-average线
        plt.axhline(y=avg_precision["macro"], color='navy', linestyle=':', linewidth=3,
                   label=f'Macro-average (AP = {avg_precision["macro"]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Multi-class Precision-Recall Curves - Fold {fold+1} ({fold_name.title()}) - MLP ({self.aggregation_method.upper()})')
        plt.legend(loc="lower left", fontsize='small')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        roc_prc_path = fold_dir / f"fold_{fold+1}_{fold_name}_roc_prc_curves.png"
        plt.savefig(roc_prc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细的AUC和AP数据
        auc_ap_data = {
            'class_specific': {},
            'macro_averages': {
                'roc_auc_macro': float(roc_auc["macro"]),
                'average_precision_macro': float(avg_precision["macro"])
            }
        }
        
        for i in range(n_classes):
            auc_ap_data['class_specific'][self.class_names[i]] = {
                'roc_auc': float(roc_auc[i]),
                'average_precision': float(avg_precision[i])
            }
        
        auc_ap_path = fold_dir / f"fold_{fold+1}_{fold_name}_auc_ap_scores.json"
        with open(auc_ap_path, 'w', encoding='utf-8') as f:
            json.dump(auc_ap_data, f, indent=2, ensure_ascii=False)
        
        # 更新metrics
        metrics['roc_auc_macro'] = float(roc_auc["macro"])
        metrics['average_precision_macro'] = float(avg_precision["macro"])
        metrics['roc_auc_per_class'] = {self.class_names[i]: float(roc_auc[i]) for i in range(n_classes)}
        metrics['average_precision_per_class'] = {self.class_names[i]: float(avg_precision[i]) for i in range(n_classes)}
        
        print(f"  ROC AUC (macro): {roc_auc['macro']:.4f}, Average Precision (macro): {avg_precision['macro']:.4f}")
    
    def train_kfold(self, k=5, num_epochs=100, batch_size=32, test_ratio=0.2, hidden_dims=[512, 256, 128]):
        """
        K折交叉验证训练
        """
        
        # 分离独立测试集
        all_indices = list(range(len(self.dataset)))
        all_labels = [self.dataset.data_df.iloc[i]['encoded_label'] for i in all_indices]
        
        from sklearn.model_selection import train_test_split
        train_val_indices, self.final_test_indices = train_test_split(
            all_indices, test_size=test_ratio, 
            stratify=all_labels, random_state=42
        )
        
        train_val_labels = [all_labels[i] for i in train_val_indices]
        
        print(f"数据分割:")
        print(f"  训练+验证集: {len(train_val_indices)} 样本")
        print(f"  独立测试集: {len(self.final_test_indices)} 样本")
        
        # K折分割
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []
        all_fold_details = []
        
        for fold, (train_rel_idx, val_rel_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
            train_idx = [train_val_indices[i] for i in train_rel_idx]
            val_idx = [train_val_indices[i] for i in val_rel_idx]
            
            print(f"\n{'='*20} Fold {fold+1}/{k} - MLP ({self.aggregation_method.upper()}) {'='*20}")
            
            # 创建数据加载器
            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            val_subset = torch.utils.data.Subset(self.dataset, val_idx)
            
            # 设置训练模式
            self.dataset.set_training_mode(True)
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2
            )
            
            self.dataset.set_training_mode(False)
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2
            )
            
            # 创建模型
            model = self.create_model(hidden_dims=hidden_dims)
            
            # 计算类别权重
            train_labels = [all_labels[i] for i in train_idx]
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(train_labels), 
                y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
            
            # 训练循环
            best_val_f1 = 0
            patience = 0
            max_patience = 15
            
            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                train_loss = 0
                num_batches = 0
                
                for features, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    logits = model(features)
                    loss = criterion(logits, labels)
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                
                # 验证阶段
                model.eval()
                val_predictions = []
                val_true = []
                val_probabilities = []
                
                with torch.no_grad():
                    for features, labels, _ in val_loader:
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        
                        logits = model(features)
                        
                        # 获取预测概率
                        probabilities = F.softmax(logits, dim=1)
                        _, predicted = logits.max(1)
                        
                        val_predictions.extend(predicted.cpu().numpy())
                        val_true.extend(labels.cpu().numpy())
                        val_probabilities.extend(probabilities.cpu().numpy())
                
                # 计算指标
                val_f1 = f1_score(val_true, val_predictions, average='weighted')
                scheduler.step(val_f1)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss/num_batches:.4f}, Val F1: {val_f1:.4f}")
                
                # 早停
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_predictions = val_predictions.copy()
                    best_val_true = val_true.copy()
                    best_val_probabilities = [prob.copy() for prob in val_probabilities]
                    patience = 0
                    # 保存模型
                    model_save_path = self.results_dir / f'mlp_{self.aggregation_method}_fold_{fold}.pth'
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience += 1
                    if patience >= max_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                # 内存清理
                if (epoch + 1) % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 保存该折结果（包含ROC/PRC曲线）
            fold_report, fold_cm, fold_metrics = self.save_fold_results(
                fold, best_val_true, best_val_predictions, "validation", 
                y_pred_proba=np.array(best_val_probabilities)
            )
            
            fold_results.append(best_val_f1)
            all_fold_details.append(fold_metrics)
            print(f"Fold {fold+1} 最佳F1分数: {best_val_f1:.4f}")
            
            # 清理内存
            del model, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 输出结果
        mean_f1 = np.mean(fold_results)
        std_f1 = np.std(fold_results)
        
        print(f"\n{'='*50}")
        print(f"MLP ({self.aggregation_method.upper()}) K折交叉验证结果:")
        print(f"平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"各折结果: {fold_results}")
        print(f"详细结果已保存到: {self.results_dir}")
        
        return fold_results
    
    def evaluate_final_test(self, hidden_dims=[512, 256, 128]):
        """在预先分离的独立测试集上评估最终性能"""
        print(f"\n{'='*20} 最终测试集评估 - MLP ({self.aggregation_method.upper()}) {'='*20}")
        
        # 检查是否已经分离了测试集
        if not hasattr(self, 'final_test_indices'):
            print("错误: 请先运行 train_kfold 方法来分离测试集")
            return None
        
        test_idx = self.final_test_indices
        print(f"测试集大小: {len(test_idx)} 样本")
        
        # 创建测试数据加载器
        test_subset = torch.utils.data.Subset(self.dataset, test_idx)
        self.dataset.set_training_mode(False)
        test_loader = DataLoader(
            test_subset, 
            batch_size=32,
            shuffle=False,
            num_workers=2
        )
        
        # 加载所有fold的模型进行集成预测
        ensemble_predictions = []
        test_true = []
        test_filenames = []
        
        # 收集所有测试样本的真实标签和文件名
        for features, labels, filenames in test_loader:
            test_true.extend(labels.numpy())
            test_filenames.extend(filenames)
        
        # 对每个fold的模型进行预测
        fold_predictions = []
        all_test_probabilities = []
        
        for fold in range(5):  # 假设使用5折
            model_path = self.results_dir / f'mlp_{self.aggregation_method}_fold_{fold}.pth'
            if model_path.exists():
                print(f"加载模型: {model_path}")
                
                # 创建模型
                model = self.create_model(hidden_dims=hidden_dims)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                fold_preds = []
                fold_probs = []
                
                with torch.no_grad():
                    for features, labels, filenames in test_loader:
                        features = features.to(self.device)
                        
                        logits = model(features)
                        
                        # 获取概率和预测
                        probabilities = F.softmax(logits, dim=1)
                        _, predicted = logits.max(1)
                        
                        fold_preds.extend(predicted.cpu().numpy())
                        fold_probs.extend(probabilities.cpu().numpy())
                
                fold_predictions.append(fold_preds)
                ensemble_predictions.append(np.array(fold_probs))
                
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算集成预测概率（用于ROC/PRC）
        if ensemble_predictions:
            ensemble_test_probs = np.mean(ensemble_predictions, axis=0)
            all_test_probabilities = ensemble_test_probs
        
        if not fold_predictions:
            print("警告: 没有找到保存的模型，无法进行测试集评估")
            return None
        
        # 集成预测（多数投票或平均概率）
        if len(ensemble_predictions) > 1:
            # 平均概率
            mean_probabilities = np.mean(ensemble_predictions, axis=0)
            final_predictions = np.argmax(mean_probabilities, axis=1)
        else:
            # 只有一个fold的结果
            final_predictions = fold_predictions[0]
        
        # 保存测试集结果（包含ROC/PRC曲线）
        test_report, test_cm, test_metrics = self.save_fold_results(
            -1, test_true, final_predictions, "final_test",
            y_pred_proba=all_test_probabilities if len(all_test_probabilities) > 0 else None
        )
        
        # 保存测试集详细信息
        test_details = pd.DataFrame({
            'filename': test_filenames,
            'true_label_idx': test_true,
            'predicted_label_idx': final_predictions,
            'true_label': [self.class_names[i] for i in test_true],
            'predicted_label': [self.class_names[i] for i in final_predictions],
            'correct': np.array(test_true) == np.array(final_predictions)
        })
        
        # 如果有多个fold，保存每个fold的预测结果
        if len(fold_predictions) > 1:
            for fold_idx, fold_preds in enumerate(fold_predictions):
                test_details[f'fold_{fold_idx+1}_prediction'] = [self.class_names[i] for i in fold_preds]
        
        test_details_path = self.results_dir / 'final_test_detailed_predictions.csv'
        test_details.to_csv(test_details_path, index=False)
        
        # 保存总体结果
        overall_results = {
            'experiment_summary': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_type': f'MLP_Classifier_{self.aggregation_method}',
                'aggregation_method': self.aggregation_method,
                'total_folds': len(fold_predictions) if fold_predictions else 0
            },
            'final_test_results': test_metrics,
            'test_ensemble_info': {
                'num_models_used': len(fold_predictions),
                'ensemble_method': 'average_probabilities' if len(fold_predictions) > 1 else 'single_model'
            }
        }
        
        # 保存测试集结果
        test_results_path = self.results_dir / 'final_test_results.json'
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        
        # 创建结果摘要
        summary_text = f"""
=== MLP ({self.aggregation_method.upper()}) 最终测试结果摘要 ===
实验时间: {overall_results['experiment_summary']['timestamp']}
模型类型: {overall_results['experiment_summary']['model_type']}

=== 独立测试集结果 ===
测试集大小: {len(test_true)} 样本
集成模型数: {len(fold_predictions)} 个
测试F1分数: {test_metrics['f1_weighted']:.4f}
测试准确率: {test_metrics['accuracy']:.4f}
测试F1宏平均: {test_metrics['f1_macro']:.4f}

=== 每类别详细结果 ===
"""
        
        # 添加每类别的详细结果
        for class_name in self.class_names:
            class_idx = self.dataset.label_encoder.transform([class_name])[0]
            true_count = np.sum(np.array(test_true) == class_idx)
            pred_count = np.sum(np.array(final_predictions) == class_idx)
            correct_count = np.sum((np.array(test_true) == class_idx) & 
                                 (np.array(final_predictions) == class_idx))
            
            if true_count > 0:
                recall = correct_count / true_count
                precision = correct_count / pred_count if pred_count > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                summary_text += f"{class_name}: 真实={true_count}, 预测={pred_count}, 正确={correct_count}, "
                summary_text += f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n"
        
        summary_text += f"\n结果文件保存在: {self.results_dir}\n"
        summary_text += f"详细预测结果: final_test_detailed_predictions.csv\n"
        summary_text += f"模型文件: mlp_{self.aggregation_method}_fold_*.pth\n"
        
        # 保存摘要
        summary_path = self.results_dir / 'final_test_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(summary_text)
        return test_metrics

def main():
    """主函数"""
    
    # 路径设置 - 与MIL.py保持一致
    csv_file = 'data/wsi_feature_labels.csv'
    feature_dir = 'data/WSI/features/h5_files'
    
    # 检查文件存在性
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件 {csv_file} 不存在!")
        return
    
    if not os.path.exists(feature_dir):
        print(f"错误: 特征目录 {feature_dir} 不存在!")
        return
    
    print("\n" + "="*60)
    print("MLP分类器 - 基于CLAM特征")
    print("="*60)
    
    # 测试不同的特征聚合方法
    aggregation_methods = ['mean', 'max', 'mean_max']
    
    for method in aggregation_methods:
        print(f"\n{'='*20} 测试 {method.upper()} 聚合方法 {'='*20}")
        
        try:
            # 创建分类器
            classifier = MLPClassifier(csv_file, feature_dir, aggregation_method=method)
            
            # 训练
            fold_results = classifier.train_kfold(
                k=5, 
                num_epochs=100,
                batch_size=32,
                test_ratio=0.2,
                hidden_dims=[512, 256, 128]
            )
            
            print(f"\n✅ MLP ({method.upper()}) 交叉验证训练完成!")
            
            # 在独立测试集上评估
            print(f"\n{'='*30} 独立测试集评估 {'='*30}")
            final_test_results = classifier.evaluate_final_test(hidden_dims=[512, 256, 128])
            
            if final_test_results:
                print(f"\n🎯 MLP ({method.upper()}) 最终测试结果:")
                print(f"   测试F1分数: {final_test_results['f1_weighted']:.4f}")
                print(f"   测试准确率: {final_test_results['accuracy']:.4f}")
            
            print(f"\n📁 完整结果保存在: {classifier.results_dir}")
            print(f"📊 包含模型文件: mlp_{method}_fold_*.pth")
            
        except Exception as e:
            print(f"❌ MLP ({method.upper()}) 训练失败: {e}")
            continue
    
    print(f"\n🎉 所有MLP聚合方法测试完成!")
    print(f"📊 查看各自results目录获取详细结果")

if __name__ == "__main__":
    main()
