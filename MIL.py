#!/usr/bin/env python3
"""
MIL Aggregation Classifier - Based on CLAM Feature Dataset
Implements multiple MIL aggregation methods: Attention-MIL, DSMIL, TransMIL
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
    MIL WSI Dataset - Maintains original patch-level features without aggregation
    Consistent data loading method with lightweight_wsi_mlp_classifier.py
    """
    
    def __init__(self, csv_file, feature_dir, sample_types=['01', '11'], 
                 max_patches=2000, enable_augmentation=False):
        """
        Args:
            csv_file: CSV file path
            feature_dir: Feature file directory
            sample_types: Sample types
            max_patches: Maximum number of patches, for memory control
            enable_augmentation: Whether to enable data augmentation
        """
        self.data_df = pd.read_csv(csv_file, dtype={'Sample_Type_Code': str})
        self.feature_dir = Path(feature_dir)
        self.max_patches = max_patches
        self.enable_augmentation = enable_augmentation
        self.training = False
        
        # Data filtering - consistent with original code
        self.data_df = self.data_df[self.data_df['Sample_Type_Code'].isin(sample_types)]
        self.data_df = self.data_df[~self.data_df['Label'].isin(['Unknown', 'Metastatic'])]
        self.data_df = self.data_df.reset_index(drop=True)
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        self.data_df['encoded_label'] = self.label_encoder.fit_transform(self.data_df['Label'])
        
        # Check data integrity
        self._check_data_integrity()
        
        # Precompute feature dimension
        self.feature_dim = self._get_feature_dimension()
        
        print(f"MIL dataset initialization completed:")
        print(f"  Number of samples: {len(self.data_df)}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Maximum patches: {self.max_patches}")
        self._print_label_distribution()
    
    def _check_data_integrity(self):
        """Check data integrity"""
        missing_files = []
        print("Checking data file integrity...")
        
        for idx in tqdm(range(min(len(self.data_df), 10)), desc="Sampling check"):
            filename = self.data_df.iloc[idx]['Filename']
            filepath = self.feature_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"Warning: Found {len(missing_files)} missing files")
        else:
            print("✅ Data file integrity check passed")
    
    def _get_feature_dimension(self):
        """Get feature dimension"""
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
                
                # Process feature shape
                if features.ndim == 3:
                    features = features.squeeze(0)
                
                if features.ndim == 2:
                    return features.shape[1]  # Return feature dimension
                else:
                    return len(features)
                    
            except Exception as e:
                print(f"Skipping file {filepath}: {e}")
                continue
        
        return 1024  # Default CLAM feature dimension
    
    def _print_label_distribution(self):
        """Print label distribution"""
        print("Label distribution:")
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
                # Consistent feature loading method with original code
                if 'features' in f.keys():
                    features = f['features'][:]
                elif 'feats' in f.keys():
                    features = f['feats'][:]
                else:
                    key = list(f.keys())[0]
                    features = f[key][:]
            
            # Feature processing
            if features.ndim == 3:
                features = features.squeeze(0)
            
            # For MIL, we need to maintain patch-level features
            if features.ndim == 1:
                # If it's 1D feature, reshape into single patch
                features = features.reshape(1, -1)
            
            # Limit patch count to control memory
            if features.shape[0] > self.max_patches:
                # Random sampling or select first N
                if self.training and self.enable_augmentation:
                    # Random sampling during training
                    indices = np.random.choice(features.shape[0], self.max_patches, replace=False)
                    features = features[indices]
                else:
                    # Select first N during validation/testing
                    features = features[:self.max_patches]
            
            # Data augmentation (only during training)
            if self.enable_augmentation and self.training:
                features = self._augment_patches(features)
            
            # Convert to tensor
            features = torch.FloatTensor(features)  # [num_patches, feature_dim]
            label = torch.LongTensor([row['encoded_label']])[0]
            
            return features, label, filename
            
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            # Return default features
            zero_features = torch.zeros(1, self.feature_dim)
            label = torch.LongTensor([row['encoded_label']])[0]
            return zero_features, label, filename
    
    def _augment_patches(self, features):
        """Patch-level data augmentation"""
        if np.random.rand() < 0.3:
            # Add small amount of noise
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise
        
        if np.random.rand() < 0.2:
            # Patch-level dropout
            num_patches = features.shape[0]
            keep_ratio = 0.9
            keep_patches = int(num_patches * keep_ratio)
            if keep_patches > 0:
                indices = np.random.choice(num_patches, keep_patches, replace=False)
                features = features[indices]
        
        return features
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training = training

# ===============================
# MIL Aggregation Method Implementations
# ===============================

class AttentionMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al.)
    Uses attention mechanism to aggregate patch features
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=5, dropout=0.25):
        super(AttentionMIL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature transformation layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_patches, input_dim] or [num_patches, input_dim]
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_patches, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, num_patches, _ = x.shape
        
        # Feature extraction
        h = self.feature_extractor(x)  # [batch_size, num_patches, hidden_dim]
        
        # Calculate attention weights
        attention_weights = self.attention(h)  # [batch_size, num_patches, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # softmax normalization
        
        # Weighted aggregation
        aggregated_features = torch.sum(attention_weights * h, dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(aggregated_features)  # [batch_size, num_classes]
        
        return logits, attention_weights

class DSMIL(nn.Module):
    """
    Dual-Stream MIL (Li et al.)
    Uses dual-stream attention mechanism
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=5, dropout=0.25):
        super(DSMIL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Instance-level classifier
        self.instance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Bag-level attention
        self.bag_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Bag-level classifier
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_patches, input_dim] or [num_patches, input_dim]
        Returns:
            bag_logits: [batch_size, num_classes]
            instance_logits: [batch_size, num_patches, num_classes]
            attention_weights: [batch_size, num_patches, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, num_patches, _ = x.shape
        
        # Feature extraction
        h = self.feature_extractor(x)  # [batch_size, num_patches, hidden_dim]
        
        # Instance-level prediction
        instance_logits = self.instance_classifier(h)  # [batch_size, num_patches, num_classes]
        
        # Attention weight calculation
        attention_weights = self.bag_attention(h)  # [batch_size, num_patches, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Bag-level feature aggregation
        bag_features = torch.sum(attention_weights * h, dim=1)  # [batch_size, hidden_dim]
        
        # Bag-level prediction
        bag_logits = self.bag_classifier(bag_features)  # [batch_size, num_classes]
        
        return bag_logits, instance_logits, attention_weights

class TransMIL(nn.Module):
    """
    Transformer-based MIL
    Uses Transformer for feature aggregation
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_heads=8, 
                 num_layers=2, num_classes=5, dropout=0.25):
        super(TransMIL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Position encoding (optional)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Aggregation method: use CLS token or global average pooling
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Classifier
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
            x: [batch_size, num_patches, input_dim] or [num_patches, input_dim]
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: Transformer attention weights
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, num_patches, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, num_patches, hidden_dim]
        
        # Add position encoding
        if num_patches <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :num_patches, :]
        
        # Add CLS token (if used)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_patches+1, hidden_dim]
        
        # Transformer encoding
        encoded = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # Feature aggregation
        if self.use_cls_token:
            # Use CLS token
            aggregated_features = encoded[:, 0, :]  # [batch_size, hidden_dim]
        else:
            # Global average pooling
            aggregated_features = torch.mean(encoded, dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(aggregated_features)  # [batch_size, num_classes]
        
        # Attention weights (simplified version, returns average attention of last layer)
        attention_weights = None  # Can get detailed attention weights through hooks
        
        return logits, attention_weights

# ===============================
# MIL Classifier Main Class
# ===============================

class MILClassifier:
    """
    MIL Classifier Main Class
    Supports multiple MIL aggregation methods
    """
    
    def __init__(self, csv_file, feature_dir, mil_method='attention', device=None):
        self.csv_file = csv_file
        self.feature_dir = feature_dir
        self.mil_method = mil_method
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"MIL method: {mil_method}")
        
        # Create dataset
        self.dataset = MILWSIDataset(
            csv_file, feature_dir, 
            max_patches=1000,  # Control memory usage
            enable_augmentation=True
        )
        
        self.num_classes = len(self.dataset.label_encoder.classes_)
        self.class_names = self.dataset.label_encoder.classes_
        self.feature_dim = self.dataset.feature_dim
        
        # Create results save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results_mil_{mil_method}_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")
        
        # Save experiment configuration
        self.save_experiment_config()
    
    def save_experiment_config(self):
        """Save experiment configuration information"""
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
        
        # Add class distribution information
        for label_name in self.class_names:
            count = np.sum(self.dataset.data_df['encoded_label'] == 
                          self.dataset.label_encoder.transform([label_name])[0])
            config['class_distribution'][label_name] = {
                'count': int(count),
                'percentage': float(count / len(self.dataset) * 100)
            }
        
        # Save configuration
        config_path = self.results_dir / 'experiment_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment configuration saved: {config_path}")
    
    def create_model(self):
        """Create MIL model based on specified method"""
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
            raise ValueError(f"Unsupported MIL method: {self.mil_method}")
        
        return model.to(self.device)
    
    def collate_fn(self, batch):
        """
        Custom collate function to handle different sized patch collections
        """
        features_list = []
        labels_list = []
        filenames_list = []
        
        for features, label, filename in batch:
            features_list.append(features)
            labels_list.append(label)
            filenames_list.append(filename)
        
        # Labels can be directly stacked
        labels = torch.stack(labels_list)
        
        return features_list, labels, filenames_list
    
    def save_fold_results(self, fold, y_true, y_pred, fold_name="validation", y_pred_proba=None):
        """Save results for each fold"""
        fold_dir = self.results_dir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Save classification report
        report_path = fold_dir / f"fold_{fold+1}_{fold_name}_classification_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(class_report, f, indent=2, ensure_ascii=False)
        
        report_df = pd.DataFrame(class_report).transpose()
        report_csv_path = fold_dir / f"fold_{fold+1}_{fold_name}_classification_report.csv"
        report_df.to_csv(report_csv_path, index=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_path = fold_dir / f"fold_{fold+1}_{fold_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True)
        
        # Plot confusion matrix
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
        
        # Calculate detailed metrics
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
        
        # Calculate and plot AUROC/AUPRC (if prediction probabilities provided)
        if y_pred_proba is not None:
            self._plot_roc_prc_curves(fold, y_true, y_pred_proba, fold_name, fold_dir, metrics)
        
        return class_report, cm, metrics
    
    def _plot_roc_prc_curves(self, fold, y_true, y_pred_proba, fold_name, fold_dir, metrics):
        """Plot ROC and PRC curves"""
        try:
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba)
            
            # For multi-class problems, need to binarize
            n_classes = len(self.class_names)
            
            if n_classes == 2:
                # Binary classification case
                self._plot_binary_roc_prc(fold, y_true, y_pred_proba, fold_name, fold_dir, metrics)
            else:
                # Multi-class case
                self._plot_multiclass_roc_prc(fold, y_true, y_pred_proba, fold_name, fold_dir, metrics)
                
        except Exception as e:
            print(f"Error plotting ROC/PRC curves: {e}")
    
    def _plot_binary_roc_prc(self, fold, y_true, y_pred_proba, fold_name, fold_dir, metrics):
        """Plot ROC and PRC curves for binary classification"""
        # Use positive class probability
        y_scores = y_pred_proba[:, 1]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # PRC curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Plot ROC curve
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
        
        # Plot PRC curve
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
        
        # Save image
        roc_prc_path = fold_dir / f"fold_{fold+1}_{fold_name}_roc_prc_curves.png"
        plt.savefig(roc_prc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Update metrics
        metrics['roc_auc'] = float(roc_auc)
        metrics['average_precision'] = float(avg_precision)
        
        print(f"  ROC AUC: {roc_auc:.4f}, Average Precision: {avg_precision:.4f}")
    
    def _plot_multiclass_roc_prc(self, fold, y_true, y_pred_proba, fold_name, fold_dir, metrics):
        """Plot ROC and PRC curves for multi-class classification"""
        n_classes = len(self.class_names)
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Calculate ROC and PRC for each class
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
        
        # Calculate macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Calculate macro-average precision
        avg_precision["macro"] = np.mean([avg_precision[i] for i in range(n_classes)])
        
        # Plot ROC curves
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        # Plot ROC curve for each class
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot macro-average ROC curve
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
        
        # Plot PRC curves
        plt.subplot(1, 2, 2)
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AP = {avg_precision[i]:.3f})')
        
        # Plot macro-average line
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
        
        # Save image
        roc_prc_path = fold_dir / f"fold_{fold+1}_{fold_name}_roc_prc_curves.png"
        plt.savefig(roc_prc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed AUC and AP data
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
        
        # Update metrics
        metrics['roc_auc_macro'] = float(roc_auc["macro"])
        metrics['average_precision_macro'] = float(avg_precision["macro"])
        metrics['roc_auc_per_class'] = {self.class_names[i]: float(roc_auc[i]) for i in range(n_classes)}
        metrics['average_precision_per_class'] = {self.class_names[i]: float(avg_precision[i]) for i in range(n_classes)}
        
        print(f"  ROC AUC (macro): {roc_auc['macro']:.4f}, Average Precision (macro): {avg_precision['macro']:.4f}")
    
    def train_kfold(self, k=5, num_epochs=100, batch_size=4, test_ratio=0.2):
        """
        K-fold cross-validation training
        Note: Due to MIL memory requirements, batch_size is set small
        """
        
        # Separate independent test set
        all_indices = list(range(len(self.dataset)))
        all_labels = [self.dataset.data_df.iloc[i]['encoded_label'] for i in all_indices]
        
        from sklearn.model_selection import train_test_split
        train_val_indices, self.final_test_indices = train_test_split(
            all_indices, test_size=test_ratio, 
            stratify=all_labels, random_state=42
        )
        
        train_val_labels = [all_labels[i] for i in train_val_indices]
        
        print(f"Data split:")
        print(f"  Training+Validation set: {len(train_val_indices)} samples")
        print(f"  Independent test set: {len(self.final_test_indices)} samples")
        
        # K-fold split
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []
        all_fold_details = []
        
        for fold, (train_rel_idx, val_rel_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
            train_idx = [train_val_indices[i] for i in train_rel_idx]
            val_idx = [train_val_indices[i] for i in val_rel_idx]
            
            print(f"\n{'='*20} Fold {fold+1}/{k} - {self.mil_method.upper()} {'='*20}")
            
            # Create data loaders
            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            val_subset = torch.utils.data.Subset(self.dataset, val_idx)
            
            # Set training mode
            self.dataset.set_training_mode(True)
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=0  # MIL usually set to 0 to avoid multi-process issues
            )
            
            self.dataset.set_training_mode(False)
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=0
            )
            
            # Create model
            model = self.create_model()
            
            # Calculate class weights
            train_labels = [all_labels[i] for i in train_idx]
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(train_labels), 
                y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
            
            # Training loop
            best_val_f1 = 0
            patience = 0
            max_patience = 15
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0
                num_batches = 0
                
                for batch_features, batch_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                    optimizer.zero_grad()
                    batch_loss = 0
                    
                    # Process each sample in batch (due to different patch counts)
                    for features, label in zip(batch_features, batch_labels):
                        features = features.to(self.device)
                        label = label.to(self.device).unsqueeze(0)
                        
                        # Forward propagation
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
                
                # Validation phase
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
                            
                            # Get prediction probabilities
                            probabilities = F.softmax(bag_logits, dim=1)
                            _, predicted = bag_logits.max(1)
                            
                            val_predictions.append(predicted.cpu().item())
                            val_true.append(label.item())
                            val_probabilities.append(probabilities.cpu().numpy())
                
                # Calculate metrics
                val_f1 = f1_score(val_true, val_predictions, average='weighted')
                scheduler.step(val_f1)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss/num_batches:.4f}, Val F1: {val_f1:.4f}")
                
                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_predictions = val_predictions.copy()
                    best_val_true = val_true.copy()
                    best_val_probabilities = [prob.copy() for prob in val_probabilities]
                    patience = 0
                    # Save model to results folder
                    model_save_path = self.results_dir / f'mil_{self.mil_method}_fold_{fold}.pth'
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience += 1
                    if patience >= max_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Memory cleanup
                if (epoch + 1) % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Save fold results (including ROC/PRC curves)
            fold_report, fold_cm, fold_metrics = self.save_fold_results(
                fold, best_val_true, best_val_predictions, "validation", 
                y_pred_proba=np.array(best_val_probabilities).squeeze()
            )
            
            fold_results.append(best_val_f1)
            all_fold_details.append(fold_metrics)
            print(f"Fold {fold+1} best F1 score: {best_val_f1:.4f}")
            
            # Cleanup memory
            del model, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Output results
        mean_f1 = np.mean(fold_results)
        std_f1 = np.std(fold_results)
        
        print(f"\n{'='*50}")
        print(f"{self.mil_method.upper()} MIL K-fold cross-validation results:")
        print(f"Average F1 score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Fold results: {fold_results}")
        print(f"Detailed results saved to: {self.results_dir}")
        
        return fold_results
    
    def evaluate_final_test(self):
        """Evaluate final performance on pre-separated independent test set"""
        print(f"\n{'='*20} Final Test Set Evaluation - {self.mil_method.upper()} {'='*20}")
        
        # Check if test set has been separated
        if not hasattr(self, 'final_test_indices'):
            print("Error: Please run train_kfold method first to separate test set")
            return None
        
        test_idx = self.final_test_indices
        print(f"Test set size: {len(test_idx)} samples")
        
        # Create test data loader
        test_subset = torch.utils.data.Subset(self.dataset, test_idx)
        self.dataset.set_training_mode(False)
        test_loader = DataLoader(
            test_subset, 
            batch_size=1,  # Use batch_size=1 during testing
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        
        # Load all fold models for ensemble prediction
        ensemble_predictions = []
        test_true = []
        test_filenames = []
        
        # Collect true labels and filenames of all test samples
        for batch_features, batch_labels, batch_filenames in test_loader:
            for label, filename in zip(batch_labels, batch_filenames):
                test_true.append(label.item())
                test_filenames.append(filename)
        
        # Make predictions for each fold model
        fold_predictions = []
        all_test_probabilities = []  # For saving test set prediction probabilities
        
        for fold in range(5):  # Assuming 5 folds
            model_path = self.results_dir / f'mil_{self.mil_method}_fold_{fold}.pth'
            if model_path.exists():
                print(f"Loading model: {model_path}")
                
                # Create model
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
                            
                            # Get probabilities and predictions
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
        
        # Calculate ensemble prediction probabilities (for ROC/PRC)
        if ensemble_predictions:
            ensemble_test_probs = np.mean(ensemble_predictions, axis=0).squeeze()
            all_test_probabilities = ensemble_test_probs
        
        if not fold_predictions:
            print("Warning: No saved models found, cannot perform test set evaluation")
            return None
        
        # Ensemble prediction (majority voting or average probability)
        if len(ensemble_predictions) > 1:
            # Average probabilities
            mean_probabilities = np.mean(ensemble_predictions, axis=0)
            final_predictions = np.argmax(mean_probabilities, axis=2).flatten()
        else:
            # Only one fold result
            final_predictions = fold_predictions[0]
        
        # Save test set results (including ROC/PRC curves)
        test_report, test_cm, test_metrics = self.save_fold_results(
            -1, test_true, final_predictions, "final_test",
            y_pred_proba=all_test_probabilities if len(all_test_probabilities) > 0 else None
        )
        
        # Save test set detailed information
        test_details = pd.DataFrame({
            'filename': test_filenames,
            'true_label_idx': test_true,
            'predicted_label_idx': final_predictions,
            'true_label': [self.class_names[i] for i in test_true],
            'predicted_label': [self.class_names[i] for i in final_predictions],
            'correct': np.array(test_true) == np.array(final_predictions)
        })
        
        # If multiple folds, save each fold's prediction results
        if len(fold_predictions) > 1:
            for fold_idx, fold_preds in enumerate(fold_predictions):
                test_details[f'fold_{fold_idx+1}_prediction'] = [self.class_names[i] for i in fold_preds]
        
        test_details_path = self.results_dir / 'final_test_detailed_predictions.csv'
        test_details.to_csv(test_details_path, index=False)
        
        # Save overall results (including cross-validation and test results)
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
        
        # Save test set results
        test_results_path = self.results_dir / 'final_test_results.json'
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        
        # Create results summary
        summary_text = f"""
=== {self.mil_method.upper()} MIL Final Test Results Summary ===
Experiment Time: {overall_results['experiment_summary']['timestamp']}
Model Type: {overall_results['experiment_summary']['model_type']}

=== Independent Test Set Results ===
Test Set Size: {len(test_true)} samples
Ensemble Models: {len(fold_predictions)}
Test F1 Score: {test_metrics['f1_weighted']:.4f}
Test Accuracy: {test_metrics['accuracy']:.4f}
Test F1 Macro Average: {test_metrics['f1_macro']:.4f}

=== Per-Class Detailed Results ===
"""
        
        # Add per-class detailed results
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
                
                summary_text += f"{class_name}: True={true_count}, Predicted={pred_count}, Correct={correct_count}, "
                summary_text += f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n"
        
        summary_text += f"\nResult files saved to: {self.results_dir}\n"
        summary_text += f"Detailed prediction results: final_test_detailed_predictions.csv\n"
        summary_text += f"Model files: mil_{self.mil_method}_fold_*.pth\n"
        
        # Save summary
        summary_path = self.results_dir / 'final_test_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(summary_text)
        return test_metrics

def main():
    """Main function"""
    
    # Path settings - consistent with lightweight_wsi_mlp_classifier.py
    csv_file = 'data/wsi_feature_labels.csv'
    feature_dir = 'data/WSI/features/h5_files'
    
    # Check file existence
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} does not exist!")
        return
    
    if not os.path.exists(feature_dir):
        print(f"Error: Feature directory {feature_dir} does not exist!")
        return
    
    print("\n" + "="*60)
    print("MIL Aggregation Classifier - Based on CLAM Features")
    print("="*60)
    
    # Test all MIL methods
    mil_methods = ['attention', 'dsmil', 'transmil']
    
    for method in mil_methods:
        print(f"\n{'='*20} Testing {method.upper()} MIL {'='*20}")
        
        try:
            # Create classifier
            classifier = MILClassifier(csv_file, feature_dir, mil_method=method)
            
            # Train
            fold_results = classifier.train_kfold(
                k=5, 
                num_epochs=50,  # MIL usually requires more epochs
                batch_size=2,   # Use smaller batch due to memory constraints
                test_ratio=0.2
            )
            
            print(f"\n✅ {method.upper()} MIL cross-validation training completed!")
            
            # Evaluate on independent test set
            print(f"\n{'='*30} Independent Test Set Evaluation {'='*30}")
            final_test_results = classifier.evaluate_final_test()
            
            if final_test_results:
                print(f"\n🎯 {method.upper()} MIL final test results:")
                print(f"   Test F1 score: {final_test_results['f1_weighted']:.4f}")
                print(f"   Test accuracy: {final_test_results['accuracy']:.4f}")
            
            print(f"\n📁 Complete results saved to: {classifier.results_dir}")
            print(f"📊 Includes model files: mil_{method}_fold_*.pth")
            
        except Exception as e:
            print(f"❌ {method.upper()} MIL training failed: {e}")
            continue
    
    print(f"\n🎉 All MIL method testing completed!")
    print(f"📊 Check respective results directories for detailed results")

if __name__ == "__main__":
    main()