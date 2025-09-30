#!/usr/bin/env python3
"""
MIL聚合分类器 - 基于CLAM特征数据集
实现多种MIL聚合方法：Attention-MIL, DSMIL, TransMIL
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

class MILWSIDataset(Dataset):
    """
    MIL WSI数据集 - 保持原始patch-level特征，不进行聚合
    与lightweight_wsi_mlp_classifier.py保持一致的数据加载方式
    """
    
    def __init__(self, csv_file, feature_dir, sample_types=['01', '11'], 
                 max_patches=2000, enable_augmentation=False):
        """
        Args:
            csv_file: CSV文件路径
            feature_dir: 特征文件目录
            sample_types: 样本类型
            max_patches: 最大patch数量，用于内存控制
            enable_augmentation: 是否启用数据增强
        """
        self.data_df = pd.read_csv(csv_file, dtype={'Sample_Type_Code': str})
        self.feature_dir = Path(feature_dir)
        self.max_patches = max_patches
        self.enable_augmentation = enable_augmentation
        self.training = False
        
        # 数据过滤 - 与原代码保持一致
        self.data_df = self.data_df[self.data_df['Sample_Type_Code'].isin(sample_types)]
        self.data_df = self.data_df[~self.data_df['Label'].isin(['Unknown', 'Metastatic'])]
        self.data_df = self.data_df.reset_index(drop=True)
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.data_df['encoded_label'] = self.label_encoder.fit_transform(self.data_df['Label'])
        
        # 检查数据完整性
        self._check_data_integrity()
        
        # 预计算特征维度
        self.feature_dim = self._get_feature_dimension()
        
        print(f"MIL数据集初始化完成:")
        print(f"  样本数量: {len(self.data_df)}")
        print(f"  特征维度: {self.feature_dim}")
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
        """获取特征维度"""
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
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        filename = row['Filename']
        filepath = self.feature_dir / filename
        
        try:
            with h5py.File(filepath, 'r') as f:
                # 与原代码保持一致的特征加载方式
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
            
            # 对于MIL，我们需要保持patch-level特征
            if features.ndim == 1:
                # 如果是1D特征，重新reshape为单个patch
                features = features.reshape(1, -1)
            
            # 限制patch数量以控制内存
            if features.shape[0] > self.max_patches:
                # 随机采样或选择前N个
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
            
            # 转换为tensor
            features = torch.FloatTensor(features)  # [num_patches, feature_dim]
            label = torch.LongTensor([row['encoded_label']])[0]
            
            return features, label, filename
            
        except Exception as e:
            print(f"加载文件错误 {filepath}: {e}")
            # 返回默认特征
            zero_features = torch.zeros(1, self.feature_dim)
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
# MIL聚合方法实现
# ===============================

class AttentionMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al.)
    使用注意力机制聚合patch特征
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=5, dropout=0.25):
        super(AttentionMIL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 特征变换层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_patches, input_dim] 或 [num_patches, input_dim]
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_patches, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加batch维度
        
        batch_size, num_patches, _ = x.shape
        
        # 特征提取
        h = self.feature_extractor(x)  # [batch_size, num_patches, hidden_dim]
        
        # 计算注意力权重
        attention_weights = self.attention(h)  # [batch_size, num_patches, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # softmax归一化
        
        # 加权聚合
        aggregated_features = torch.sum(attention_weights * h, dim=1)  # [batch_size, hidden_dim]
        
        # 分类
        logits = self.classifier(aggregated_features)  # [batch_size, num_classes]
        
        return logits, attention_weights

class DSMIL(nn.Module):
    """
    Dual-Stream MIL (Li et al.)
    使用双流注意力机制
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=5, dropout=0.25):
        super(DSMIL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 实例级分类器
        self.instance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 包级注意力
        self.bag_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 包级分类器
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_patches, input_dim] 或 [num_patches, input_dim]
        Returns:
            bag_logits: [batch_size, num_classes]
            instance_logits: [batch_size, num_patches, num_classes]
            attention_weights: [batch_size, num_patches, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, num_patches, _ = x.shape
        
        # 特征提取
        h = self.feature_extractor(x)  # [batch_size, num_patches, hidden_dim]
        
        # 实例级预测
        instance_logits = self.instance_classifier(h)  # [batch_size, num_patches, num_classes]
        
        # 注意力权重计算
        attention_weights = self.bag_attention(h)  # [batch_size, num_patches, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 包级特征聚合
        bag_features = torch.sum(attention_weights * h, dim=1)  # [batch_size, hidden_dim]
        
        # 包级预测
        bag_logits = self.bag_classifier(bag_features)  # [batch_size, num_classes]
        
        return bag_logits, instance_logits, attention_weights

class TransMIL(nn.Module):
    """
    Transformer-based MIL
    使用Transformer进行特征聚合
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_heads=8, 
                 num_layers=2, num_classes=5, dropout=0.25):
        super(TransMIL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码（可选）
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.1)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 聚合方式：使用CLS token或全局平均池化
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_patches, input_dim] 或 [num_patches, input_dim]
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: Transformer注意力权重
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, num_patches, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, num_patches, hidden_dim]
        
        # 添加位置编码
        if num_patches <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :num_patches, :]
        
        # 添加CLS token（如果使用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_patches+1, hidden_dim]
        
        # Transformer编码
        encoded = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # 特征聚合
        if self.use_cls_token:
            # 使用CLS token
            aggregated_features = encoded[:, 0, :]  # [batch_size, hidden_dim]
        else:
            # 全局平均池化
            aggregated_features = torch.mean(encoded, dim=1)  # [batch_size, hidden_dim]
        
        # 分类
        logits = self.classifier(aggregated_features)  # [batch_size, num_classes]
        
        # 注意力权重（简化版本，返回最后一层的平均注意力）
        attention_weights = None  # 可以通过hook获取详细注意力权重
        
        return logits, attention_weights

# ===============================
# MIL分类器主类
# ===============================

class MILClassifier:
    """
    MIL分类器主类
    支持多种MIL聚合方法
    """
    
    def __init__(self, csv_file, feature_dir, mil_method='attention', device=None):
        self.csv_file = csv_file
        self.feature_dir = feature_dir
        self.mil_method = mil_method
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"MIL方法: {mil_method}")
        
        # 创建数据集
        self.dataset = MILWSIDataset(
            csv_file, feature_dir, 
            max_patches=1000,  # 控制内存使用
            enable_augmentation=True
        )
        
        self.num_classes = len(self.dataset.label_encoder.classes_)
        self.class_names = self.dataset.label_encoder.classes_
        self.feature_dim = self.dataset.feature_dim
        
        # 创建结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results_mil_{mil_method}_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        print(f"结果将保存到: {self.results_dir}")
        
        # 保存实验配置
        self.save_experiment_config()
    
    def save_experiment_config(self):
        """保存实验配置信息"""
        config = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': f'MIL_Classifier_{self.mil_method}',
            'mil_method': self.mil_method,
            'device': str(self.device),
            'num_classes': self.num_classes,
            'class_names': list(self.class_names),
            'feature_dim': self.feature_dim,
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
    
    def create_model(self):
        """根据指定方法创建MIL模型"""
        if self.mil_method == 'attention':
            model = AttentionMIL(
                input_dim=self.feature_dim,
                hidden_dim=256,
                num_classes=self.num_classes,
                dropout=0.25
            )
        elif self.mil_method == 'dsmil':
            model = DSMIL(
                input_dim=self.feature_dim,
                hidden_dim=256,
                num_classes=self.num_classes,
                dropout=0.25
            )
        elif self.mil_method == 'transmil':
            model = TransMIL(
                input_dim=self.feature_dim,
                hidden_dim=256,
                num_heads=8,
                num_layers=2,
                num_classes=self.num_classes,
                dropout=0.25
            )
        else:
            raise ValueError(f"不支持的MIL方法: {self.mil_method}")
        
        return model.to(self.device)
    
    def collate_fn(self, batch):
        """
        自定义collate函数处理不同大小的patch集合
        """
        features_list = []
        labels_list = []
        filenames_list = []
        
        for features, label, filename in batch:
            features_list.append(features)
            labels_list.append(label)
            filenames_list.append(filename)
        
        # 标签可以直接stack
        labels = torch.stack(labels_list)
        
        return features_list, labels, filenames_list
    
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
        plt.title(f'Confusion Matrix - Fold {fold+1} ({fold_name.title()}) - {self.mil_method.upper()}')
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
            'mil_method': self.mil_method,
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
        plt.title(f'ROC Curve - Fold {fold+1} ({fold_name.title()}) - {self.mil_method.upper()}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 绘制PRC曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Fold {fold+1} ({fold_name.title()}) - {self.mil_method.upper()}')
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
        plt.title(f'Multi-class ROC Curves - Fold {fold+1} ({fold_name.title()}) - {self.mil_method.upper()}')
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
        plt.title(f'Multi-class Precision-Recall Curves - Fold {fold+1} ({fold_name.title()}) - {self.mil_method.upper()}')
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
    
    def train_kfold(self, k=5, num_epochs=100, batch_size=4, test_ratio=0.2):
        """
        K折交叉验证训练
        注意：由于MIL的内存需求，batch_size设置较小
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
            
            print(f"\n{'='*20} Fold {fold+1}/{k} - {self.mil_method.upper()} {'='*20}")
            
            # 创建数据加载器
            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            val_subset = torch.utils.data.Subset(self.dataset, val_idx)
            
            # 设置训练模式
            self.dataset.set_training_mode(True)
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=0  # MIL通常设置为0避免多进程问题
            )
            
            self.dataset.set_training_mode(False)
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=0
            )
            
            # 创建模型
            model = self.create_model()
            
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
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
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
                
                for batch_features, batch_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                    optimizer.zero_grad()
                    batch_loss = 0
                    
                    # 处理batch中的每个样本（因为patch数量不同）
                    for features, label in zip(batch_features, batch_labels):
                        features = features.to(self.device)
                        label = label.to(self.device).unsqueeze(0)
                        
                        # 前向传播
                        if self.mil_method == 'dsmil':
                            bag_logits, instance_logits, attention_weights = model(features)
                            loss = criterion(bag_logits, label)
                        else:
                            logits, attention_weights = model(features)
                            loss = criterion(logits, label)
                        
                        batch_loss += loss
                    
                    batch_loss = batch_loss / len(batch_features)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += batch_loss.item()
                    num_batches += 1
                
                # 验证阶段
                model.eval()
                val_predictions = []
                val_true = []
                val_probabilities = []
                
                with torch.no_grad():
                    for batch_features, batch_labels, _ in val_loader:
                        for features, label in zip(batch_features, batch_labels):
                            features = features.to(self.device)
                            
                            if self.mil_method == 'dsmil':
                                bag_logits, _, _ = model(features)
                            else:
                                bag_logits, _ = model(features)
                            
                            # 获取预测概率
                            probabilities = F.softmax(bag_logits, dim=1)
                            _, predicted = bag_logits.max(1)
                            
                            val_predictions.append(predicted.cpu().item())
                            val_true.append(label.item())
                            val_probabilities.append(probabilities.cpu().numpy())
                
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
                    # 将模型保存到结果文件夹中
                    model_save_path = self.results_dir / f'mil_{self.mil_method}_fold_{fold}.pth'
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
                y_pred_proba=np.array(best_val_probabilities).squeeze()
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
        print(f"{self.mil_method.upper()} MIL K折交叉验证结果:")
        print(f"平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"各折结果: {fold_results}")
        print(f"详细结果已保存到: {self.results_dir}")
        
        return fold_results
    
    def evaluate_final_test(self):
        """在预先分离的独立测试集上评估最终性能"""
        print(f"\n{'='*20} 最终测试集评估 - {self.mil_method.upper()} {'='*20}")
        
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
            batch_size=1,  # 测试时使用batch_size=1
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        
        # 加载所有fold的模型进行集成预测
        ensemble_predictions = []
        test_true = []
        test_filenames = []
        
        # 收集所有测试样本的真实标签和文件名
        for batch_features, batch_labels, batch_filenames in test_loader:
            for label, filename in zip(batch_labels, batch_filenames):
                test_true.append(label.item())
                test_filenames.append(filename)
        
        # 对每个fold的模型进行预测
        fold_predictions = []
        all_test_probabilities = []  # 用于保存测试集的预测概率
        
        for fold in range(5):  # 假设使用5折
            model_path = self.results_dir / f'mil_{self.mil_method}_fold_{fold}.pth'
            if model_path.exists():
                print(f"加载模型: {model_path}")
                
                # 创建模型
                model = self.create_model()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                fold_preds = []
                fold_probs = []
                
                with torch.no_grad():
                    for batch_features, batch_labels, batch_filenames in test_loader:
                        for features in batch_features:
                            features = features.to(self.device)
                            
                            if self.mil_method == 'dsmil':
                                bag_logits, _, _ = model(features)
                            else:
                                bag_logits, _ = model(features)
                            
                            # 获取概率和预测
                            probabilities = F.softmax(bag_logits, dim=1)
                            _, predicted = bag_logits.max(1)
                            
                            fold_preds.append(predicted.cpu().item())
                            fold_probs.append(probabilities.cpu().numpy())
                
                fold_predictions.append(fold_preds)
                ensemble_predictions.append(np.array(fold_probs))
                
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算集成预测概率（用于ROC/PRC）
        if ensemble_predictions:
            ensemble_test_probs = np.mean(ensemble_predictions, axis=0).squeeze()
            all_test_probabilities = ensemble_test_probs
        
        if not fold_predictions:
            print("警告: 没有找到保存的模型，无法进行测试集评估")
            return None
        
        # 集成预测（多数投票或平均概率）
        if len(ensemble_predictions) > 1:
            # 平均概率
            mean_probabilities = np.mean(ensemble_predictions, axis=0)
            final_predictions = np.argmax(mean_probabilities, axis=2).flatten()
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
        
        # 保存总体结果（包含交叉验证和测试结果）
        overall_results = {
            'experiment_summary': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_type': f'MIL_Classifier_{self.mil_method}',
                'mil_method': self.mil_method,
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
=== {self.mil_method.upper()} MIL 最终测试结果摘要 ===
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
        summary_text += f"模型文件: mil_{self.mil_method}_fold_*.pth\n"
        
        # 保存摘要
        summary_path = self.results_dir / 'final_test_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(summary_text)
        return test_metrics

def main():
    """主函数"""
    
    # 路径设置 - 与lightweight_wsi_mlp_classifier.py保持一致
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
    print("MIL聚合分类器 - 基于CLAM特征")
    print("="*60)
    
    # 测试所有MIL方法
    mil_methods = ['attention', 'dsmil', 'transmil']
    
    for method in mil_methods:
        print(f"\n{'='*20} 测试 {method.upper()} MIL {'='*20}")
        
        try:
            # 创建分类器
            classifier = MILClassifier(csv_file, feature_dir, mil_method=method)
            
            # 训练
            fold_results = classifier.train_kfold(
                k=5, 
                num_epochs=50,  # MIL通常需要更多epoch
                batch_size=2,   # 由于内存限制，使用较小batch
                test_ratio=0.2
            )
            
            print(f"\n✅ {method.upper()} MIL 交叉验证训练完成!")
            
            # 在独立测试集上评估
            print(f"\n{'='*30} 独立测试集评估 {'='*30}")
            final_test_results = classifier.evaluate_final_test()
            
            if final_test_results:
                print(f"\n🎯 {method.upper()} MIL 最终测试结果:")
                print(f"   测试F1分数: {final_test_results['f1_weighted']:.4f}")
                print(f"   测试准确率: {final_test_results['accuracy']:.4f}")
            
            print(f"\n📁 完整结果保存在: {classifier.results_dir}")
            print(f"📊 包含模型文件: mil_{method}_fold_*.pth")
            
        except Exception as e:
            print(f"❌ {method.upper()} MIL 训练失败: {e}")
            continue
    
    print(f"\n🎉 所有MIL方法测试完成!")
    print(f"📊 查看各自results目录获取详细结果")

if __name__ == "__main__":
    main()
