"""
RDA-Net: Riemannian Domain-Adaptive Network for EEG Motor Imagery Classification
------------------------------------------------------------------------------

This script implements an end-to-end EEG motor imagery classification framework
designed for subject-dependent learning on BCI Competition IV
datasets (2a / 2b).

Core Components:
1. Riemannian Geometry Layer
   - Computes covariance matrices from EEG trials
   - Applies log-Cholesky mapping for SPD manifold projection
   - Produces compact global spatial representations

2. Lightweight CNN Encoder
   - Extracts local spatio-temporal EEG features
   - Uses depthwise separable convolutions for efficiency
   - Includes residual temporal convolution for stability

3. Sparse Transformer Encoder
   - Models long-range temporal dependencies
   - Uses window-based sparse self-attention to reduce complexity
   - Incorporates positional embeddings

4. Domain Adaptation (DANN-style, implemented but not used in this
   subject-dependent training script)
   - Gradient Reversal Layer (GRL)
   - Domain discriminator for learning subject-invariant features

5. Inter-Segment Data Augmentation
   - Generates synthetic EEG trials by segment recombination
   - Improves robustness and class balance

Training Strategies:
- Subject-dependent evaluation
- AdamW optimizer with gradient clipping
- Checkpoint selection based on minimum validation loss
- Final test evaluation performed once after training

This implementation is designed for research reproducibility and extensibility.
"""


import os
import numpy as np
import pandas as pd
import random
import datetime
import time
from utils import calMetrics, numberClassChannel, load_data_evaluate
from pandas import ExcelWriter

import torch
from torch.backends import cudnn
import math
import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True
from torchinfo import summary
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader, TensorDataset

"""
    Riemannian Geometry Feature Extractor for EEG Signals.

    This layer:
    - Computes covariance matrices from EEG trials
    - Applies regularization for numerical stability
    - Projects SPD matrices to tangent space using log-Cholesky mapping
    - Flattens the upper triangular part for compact representation

    Output:
    - A global EEG representation of shape (B, 1, emb_size)
"""
class RiemannianGeometryLayer(nn.Module):
    def __init__(self, num_channels=22, emb_size=48, reg=1e-5, schoenberg_reg=1e-3):
        super().__init__()
        self.num_channels = num_channels
        self.emb_size = emb_size
        self.reg = reg
        self.schoenberg_reg = schoenberg_reg
        
        self.proj = nn.Sequential(
            nn.Linear(num_channels * (num_channels + 1) // 2, 256),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_size)
        )

    def compute_covariance(self, x):
        B, C, T = x.shape
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (T - 1)
        
        identity = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
        cov += identity * self.reg
        
        traces = torch.diagonal(cov, dim1=1, dim2=2).sum(dim=1)
        cov += identity * traces.mean() * self.schoenberg_reg
        
        return cov

    def symmetric_matrix_log(self, cov):
        L = torch.linalg.cholesky(cov)
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
        log_diag = torch.log(diag_L.clamp(min=1e-10))
        log_cov = torch.matmul(L, torch.diag_embed(log_diag))
        return log_cov

    def forward(self, x):
        cov = self.compute_covariance(x)
        log_cov = self.symmetric_matrix_log(cov)
        
        triu_indices = torch.triu_indices(self.num_channels, self.num_channels, device=x.device)
        flattened = log_cov[:, triu_indices[0], triu_indices[1]]
        
        return self.proj(flattened).unsqueeze(1)


"""
    Lightweight CNN-based Spatio-Temporal Encoder.

    Design principles:
    - Inspired by EEGNet-style architectures
    - Depthwise separable convolutions for channel-wise filtering
    - Temporal convolutions for local dynamics
    - Residual temporal refinement for stability

    Input shape:
        (B, 1, Channels, Time)

    Output shape:
        (B, Tokens, emb_size)
"""
class LightweightCNN(nn.Module):
    def __init__(self, in_channels=22, F1=16, D=2, emb_size=48, dropout=0.5, 
                 kernel1=64, kernel2=16, pool1=8, pool2=8):
        super().__init__()
        F2 = F1 * D
        padding1 = (0, kernel1 // 2)
        padding2 = (0, kernel2 // 2)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel1), padding=padding1, bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, kernel2), padding=padding2, groups=F2, bias=False),
            nn.BatchNorm2d(F2),
            nn.Conv2d(F2, emb_size, (1, 1), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
            Rearrange('b e h w -> b (h w) e')
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(emb_size, emb_size//2, 3, padding=1, groups=emb_size//2),
            nn.BatchNorm1d(emb_size//2),
            nn.GELU(),
            nn.Conv1d(emb_size//2, emb_size, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x + self.temporal_conv(x.transpose(1,2)).transpose(1,2)
        return x


"""
    Window-based Sparse Multi-Head Self-Attention.

    Key idea:
    - Each time step attends only to a local temporal window
    - Reduces quadratic complexity of full attention
    - Preserves local temporal dependencies critical for EEG

    This is especially important for long EEG trials (e.g., 1000 time samples).
    """
class SparseAttention(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=5):
        super().__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.proj = nn.Linear(emb_size, emb_size)
        self.attn_drop = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool)
        for i in range(N):
            start = max(0, i - self.window_size)
            end = min(N, i + self.window_size + 1)
            mask[i, start:end] = False
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


"""
    Sparse Transformer Encoder Block.

    Structure:
    - Pre-norm LayerNorm
    - Sparse self-attention
    - Feed-forward network (FFN)
    - Residual connections

    Purpose:
    - Capture global temporal dependencies
    - Maintain efficiency for long EEG sequences
"""
class SparseTransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=5, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = SparseAttention(emb_size, num_heads, window_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x


"""
    Gradient Reversal Layer (GRL).

    During forward pass:
    - Acts as identity

    During backward pass:
    - Reverses gradients scaled by alpha

    This enables adversarial domain adaptation by encouraging
    domain-invariant feature learning.
"""
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

"""
    Domain Discriminator Network.

    Goal:
    - Predict whether features come from source or target domain
    - Used adversarially with GRL to enforce subject-invariant features

    Output:
    - Probability of belonging to target domain
"""
class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)



"""
    RDA-Net: Riemannian Domain-Adaptive Network for EEG Motor Imagery Classification.

    Architecture Overview:
    ---------------------------------------------------
    EEG Trial
      ├── Riemannian Geometry Layer (global spatial features)
      ├── Lightweight CNN (local spatio-temporal features)
      ├── Feature Fusion + Positional Encoding
      ├── Sparse Transformer Encoder (temporal modeling)
      ├── Global Average Pooling
      ├── Classifier (motor imagery prediction)
      └── Domain Discriminator (optional, for DA)

    Supports:
    - Standard classification
    - Adversarial domain adaptation (DANN)
"""
class HSTRDA(nn.Module):
    def __init__(self, num_classes=4, num_channels=22, emb_size=48, num_heads=8, 
                 depth=4, dropout=0.5, alpha=0.1, window_size=5, max_seq_len=500):
        super().__init__()
        self.alpha = alpha
        self.max_seq_len = max_seq_len
        
        self.riemannian = RiemannianGeometryLayer(num_channels=num_channels, emb_size=emb_size)
        self.cnn = LightweightCNN(num_channels, emb_size=emb_size, dropout=dropout)
        
        self.fusion_proj = nn.Parameter(torch.randn(emb_size, emb_size))
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, emb_size))
        
        self.transformer_blocks = nn.ModuleList([
            SparseTransformerEncoderLayer(
                emb_size, 
                num_heads, 
                window_size=window_size,
                dim_feedforward=emb_size*4,
                dropout=dropout
            ) for _ in range(depth)])
        
        self.domain_disc = DomainDiscriminator(emb_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, domain_adapt=False):
        # Remove singleton dimension for Riemannian processing
        x_riemann = x.squeeze(1)
        # Global spatial representation via Riemannian geometry
        r_feat = self.riemannian(x_riemann)
        # Local spatio-temporal features via CNN
        cnn_feat = self.cnn(x)
        
        # Feature fusion (concatenation + linear projection)
        fused = torch.cat([r_feat, cnn_feat], dim=1)
        fused = torch.matmul(fused, self.fusion_proj)

        # Add learnable positional encoding
        seq_len = fused.size(1)
        fused += self.pos_enc[:, :seq_len]
        
        # Sparse Transformer encoding
        for block in self.transformer_blocks:
            fused = block(fused)
            
        # Global representation via temporal average pooling
        global_feat = fused.mean(dim=1)
        
        domain_out = None
        if domain_adapt:
            rev_feat = GradientReversal.apply(global_feat, self.alpha)
            domain_out = self.domain_disc(rev_feat)
            
        cls_out = self.classifier(global_feat)
        
        return cls_out, domain_out


class ExP:
    def __init__(self, nsub, data_dir, result_name, 
                 epochs=1000, 
                 number_aug=3,
                 number_seg=8, 
                 gpus=[0], 
                 evaluate_mode='subject-dependent',
                 heads=8, 
                 emb_size=48,
                 depth=4, 
                 dataset_type='A',
                 validate_ratio=0.2,
                 learning_rate=0.001,
                 batch_size=64,
                 window_size=5,
                 dropout=0.5,
                 alpha=0.1,
                 da_lambda=0.5):
        self.b1 = 0.5
        self.b2 = 0.999
        self.dataset_type = dataset_type
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_epochs = epochs
        self.nSub = nsub
        self.root = data_dir
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio
        self.da_lambda = da_lambda

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = nn.CrossEntropyLoss().cuda()
        self.criterion_domain = nn.BCELoss().cuda()

        self.number_class, self.num_channels = 4, 22
        self.model = HSTRDA(
            num_classes=self.number_class,
            num_channels=self.num_channels,
            emb_size=emb_size,
            num_heads=heads,
            depth=depth,
            dropout=dropout,
            alpha=alpha,
            window_size=window_size
        ).cuda()
        self.model_filename = f'{self.result_name}/model_{self.nSub}.pth'

    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        num_per_class = self.batch_size // self.number_class
        
        for cls_idx in range(1, self.number_class+1):
            cls_mask = (label == cls_idx)
            cls_data = timg[cls_mask]
            
            if len(cls_data) == 0:
                continue
                
            cls_aug = np.zeros((num_per_class, 1, self.num_channels, 1000))
            seg_len = 1000 // self.number_seg
            
            for i in range(num_per_class):
                for seg in range(self.number_seg):
                    rand_sample = np.random.randint(0, len(cls_data))
                    start = seg * seg_len
                    end = (seg + 1) * seg_len
                    cls_aug[i, :, :, start:end] = cls_data[rand_sample, :, :, start:end]
            
            aug_data.append(cls_aug)
            aug_label.append(np.full((num_per_class,), cls_idx))
        
        if not aug_data:
            return torch.tensor([]).cuda(), torch.tensor([]).cuda()
        
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        shuffle_idx = np.random.permutation(len(aug_data))
        aug_data = aug_data[shuffle_idx]
        aug_label = aug_label[shuffle_idx]

        aug_data = torch.from_numpy(aug_data).float().cuda()
        aug_label = torch.from_numpy(aug_label - 1).long().cuda()
        return aug_data, aug_label

    def get_source_data(self):
        (train_data, train_label, test_data, test_label) = load_data_evaluate(
            self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode
        )

        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)[0]  # Fix: Extract array properly

        shuffle_idx = np.random.permutation(len(train_data))
        train_data = train_data[shuffle_idx, :, :, :]
        train_label = train_label[shuffle_idx]

        print('Train size:', train_data.shape, 'Test size:', test_data.shape)
        
        test_data = np.expand_dims(test_data, axis=1) if test_data.ndim == 3 else test_data
        test_label = np.transpose(test_label)[0]  # Fix: Extract array properly

        # Standardize
        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        
        return train_data, train_label, test_data, test_label

    def stratified_train_val_split(self, labels, val_ratio=0.2, seed=2025):
        """
        Split only the training data into train/validation subsets.
        The independent test set is not used for model selection.
        Labels are expected in 1-based format: 1, 2, ..., K.
        """
        rng = np.random.RandomState(seed)
        train_idx, val_idx = [], []

        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)

            n_val = max(1, int(len(cls_indices) * val_ratio))
            val_idx.extend(cls_indices[:n_val])
            train_idx.extend(cls_indices[n_val:])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    def evaluate_loader(self, loader, return_preds=False):
        """Evaluate a dataloader without parameter updates."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data, labels in loader:
                data = data.cuda().float()
                labels = labels.cuda().long()

                outputs, _ = self.model(data, domain_adapt=False)
                loss = self.criterion_cls(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                total_loss += loss.item() * labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                if return_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        if return_preds:
            return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds)
        return avg_loss, avg_acc

    def train(self):
        # Load data. The test set remains unseen until final evaluation.
        source_data_np, source_label_np, test_data_np, test_label_np = self.get_source_data()

        # ---------------------------------------------------------
        # 1. Create validation split from training data only
        # ---------------------------------------------------------
        train_idx, val_idx = self.stratified_train_val_split(
            source_label_np,
            val_ratio=self.validate_ratio,
            seed=2025 + self.nSub
        )

        train_data_np = source_data_np[train_idx]
        train_label_np = source_label_np[train_idx]
        val_data_np = source_data_np[val_idx]
        val_label_np = source_label_np[val_idx]

        train_dataset = TensorDataset(
            torch.from_numpy(train_data_np).float(),
            torch.from_numpy(train_label_np - 1).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_data_np).float(),
            torch.from_numpy(val_label_np - 1).long()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_data_np).float(),
            torch.from_numpy(test_label_np - 1).long()
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # ---------------------------------------------------------
        # 2. Subject-dependent training: no domain adaptation and no test-set use
        # ---------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = 300
        history = []

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            cls_loss_total = 0.0
            num_batches = 0

            for source_x, source_y in train_loader:
                source_x = source_x.cuda().float()
                source_y = source_y.cuda().long()

                # Augment training data only, never validation/test data.
                aug_x, aug_y = self.interaug(train_data_np, train_label_np)

                if len(aug_x) > 0:
                    combined_x = torch.cat([source_x, aug_x], dim=0)
                    combined_y = torch.cat([source_y, aug_y], dim=0)
                else:
                    combined_x = source_x
                    combined_y = source_y

                outputs, _ = self.model(combined_x, domain_adapt=False)
                cls_loss = self.criterion_cls(outputs, combined_y)

                self.optimizer.zero_grad()
                cls_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += cls_loss.item()
                cls_loss_total += cls_loss.item()
                num_batches += 1

            # -----------------------------------------------------
            # 3. Validation: select checkpoint by minimum validation loss
            # -----------------------------------------------------
            val_loss, val_acc = self.evaluate_loader(val_loader)

            history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss / max(num_batches, 1),
                'cls_loss': cls_loss_total / max(num_batches, 1),
                'val_loss': val_loss,
                'val_acc': val_acc * 100
            })

            print(f'Epoch [{epoch+1}/{self.n_epochs}] | '
                  f'Train Loss: {epoch_loss/max(num_batches, 1):.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_filename)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # ---------------------------------------------------------
        # 4. Final test: evaluate the independent test set only once
        # ---------------------------------------------------------
        self.model.load_state_dict(torch.load(self.model_filename))
        self.model.eval()

        test_loss, test_acc, all_labels, all_preds = self.evaluate_loader(
            test_loader,
            return_preds=True
        )

        accuracy, precison, recall, f1, kappa = calMetrics(
            np.array(all_labels),
            np.array(all_preds)
        )

        print(f'Subject {self.nSub} | Best Epoch: {best_epoch} | '
              f'Best Val Loss: {best_val_loss:.4f} | '
              f'Best Val Acc: {best_val_acc*100:.2f}% | '
              f'Final Test Acc: {accuracy:.4f} | Kappa: {kappa:.4f}')

        df_history = pd.DataFrame(history)
        return accuracy, all_labels, all_preds, best_epoch, df_history


def main(dirs,                
         evaluate_mode='subject-dependent',
         heads=8,
         emb_size=40,
         depth=6,
         dataset_type='A',
         validate_ratio=0.2,
         window_size=5,
         dropout=0.5,
         alpha=0.1,
         number_aug=3,
         number_seg=8,
         learning_rate=0.0005,
         batch_size=64,
         da_lambda=0.5):
    
    os.makedirs(dirs, exist_ok=True)
    result_metric = ExcelWriter(os.path.join(dirs, "results.xlsx"))
    results = []
    best_epochs = []
    
    for subject in range(1, 10):  # 9 subjects for BCI-IV
        print(f'\n{"="*50}')
        print(f'Training Subject {subject}')
        print(f'{"="*50}')
        
        # Set random seeds for reproducibility
        seed = 42 + subject
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        exp = ExP(
            nsub=subject,
            data_dir='/data0/Durrani/mymat_raw/',
            result_name=dirs,
            evaluate_mode=evaluate_mode,
            heads=heads,
            emb_size=emb_size,
            depth=depth,
            dataset_type=dataset_type,
            validate_ratio=validate_ratio,
            learning_rate=learning_rate,
            batch_size=batch_size,
            window_size=window_size,
            dropout=dropout,
            alpha=alpha,
            da_lambda=da_lambda,
            number_aug=number_aug,
            number_seg=number_seg
        )
        
        acc, true, pred, epoch, df_history = exp.train()
        df_history.to_excel(os.path.join(dirs, f'training_history_subject_{subject}.xlsx'), index=False)
        results.append({
            'Subject': subject,
            'Accuracy': acc,
            'Best Epoch': epoch
        })
        best_epochs.append(epoch)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_excel(result_metric, index=False)
    result_metric.close()
    
    print('\nFinal Results:')
    print(df)
    print(f'\nAverage Accuracy: {df["Accuracy"].mean():.4f}')
    #print(f'Average Best Epoch: {sum(best_epochs)/len(best_epochs):.1f}')
    print(time.asctime(time.localtime(time.time())))




if __name__ == "__main__":
    # Optimal parameters for BCI-IV dataset
    RESULT_DIR = f"./results_BCI_IV_{int(time.time())}"
    
    main(
        dirs=RESULT_DIR,
        evaluate_mode='subject-dependent',
        heads=8,
        emb_size=48,
        depth=4,
        dataset_type='A',
        validate_ratio=0.2,
        window_size=5,
        dropout=0.5,
        alpha=0.1,
        number_aug=3,
        number_seg=8,
        learning_rate=0.001,
        batch_size=64,
        da_lambda=0.5
    )
    
