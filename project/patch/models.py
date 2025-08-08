import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torchvision.models as models
import timm.models as timm_models
from fastervit import create_model
import torch.hub as hub
import torch.serialization
import argparse
import os
import yaml
import sys
import torch.nn.functional as F


from data import get_num_patches_from_config  # Import from data.py

torch.serialization.add_safe_globals([argparse.Namespace])


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=2, num_patches=2):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        for param in self.transformer.parameters():
            param.requires_grad = False
        feature_dim = self.transformer.norm.normalized_shape[0]  # 768
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.num_patches = num_patches

    def forward(self, x):
        batch_size, num_patches, C, H, W = (
            x.size()
        )  # x: [batch_size, num_patches, C, H, W]
        x = x.view(-1, C, H, W)  # [batch_size * num_patches, C, H, W]
        x = self.transformer(x)  # [batch_size * num_patches, feature_dim]
        x = self.transformer.norm(x)
        x = x.view(
            batch_size, num_patches, -1
        )  # [batch_size, num_patches, feature_dim]
        x = x.transpose(0, 1)  # [num_patches, batch_size, feature_dim]
        attn_output, _ = self.attention(x, x, x)
        x = (
            attn_output.transpose(0, 1).contiguous().view(batch_size, -1)
        )  # [batch_size, num_patches * feature_dim]
        x = self.classifier(x)
        return x


class PatchResNet(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, num_patches):
        print(f"Using PatchResNet with {num_patches} patches")
        super(PatchResNet, self).__init__()
        self.base_model = base_model
        self.layer4 = getattr(base_model, "layer4", None)  # For Grad-CAM
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.num_patches = num_patches

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x = x.view(-1, C, H, W)  # [batch_size * num_patches, C, H, W]
        x = self.base_model(x)  # [batch_size * num_patches, feature_dim]
        x = x.view(
            batch_size, num_patches * x.size(-1)
        )  # [batch_size, num_patches * feature_dim]
        x = self.classifier(x)
        return x


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self, base_model, feature_dim, num_classes, num_patches, nhead=8, num_layers=1
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        # Learnable positional encoding for each patch
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim))
        # Transformer encoder with batch_first=True and reduced dim_feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim,  # Reduced from 4 * feature_dim
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        x = self.base_model(x)
        x = x.view(batch_size, num_patches, self.feature_dim)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.classifier(x)
        return x


class TokenMixerClassifier(nn.Module):
    def __init__(
        self, base_model, feature_dim, num_classes, num_patches, nhead=4, num_layers=1
    ):
        super(TokenMixerClassifier, self).__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_patches = num_patches

        # Convolutional tokenizer to reduce patch tokens
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim // 4, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Reduced feature dimension after tokenizer
        self.reduced_dim = feature_dim // 4

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, self.reduced_dim))

        # Lightweight Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_dim,
            nhead=nhead,  # Reduced number of heads
            dim_feedforward=self.reduced_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x = x.view(-1, C, H, W)  # [batch_size * num_patches, C, H, W]

        # Extract features using CNN base model
        x = self.base_model(x)  # [batch_size * num_patches, feature_dim]

        # Reshape to apply tokenizer
        if x.dim() == 2:  # If base_model outputs flat features
            x = x.view(batch_size * num_patches, self.feature_dim, 1, 1)
        elif x.dim() == 4:  # If base_model outputs feature maps
            pass  # Already in [batch_size * num_patches, feature_dim, H', W']

        # Apply tokenizer
        x = self.tokenizer(x)  # [batch_size * num_patches, reduced_dim, 1, 1]
        x = x.view(
            batch_size, num_patches, self.reduced_dim
        )  # [batch_size, num_patches, reduced_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Apply Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, num_patches, reduced_dim]

        # Flatten and classify
        x = x.contiguous().view(
            batch_size, -1
        )  # [batch_size, num_patches * reduced_dim]
        x = self.classifier(x)
        return x


class PatchGlobalLocalClassifier(nn.Module):
    """
    Nhận đầu vào gồm các patch cục bộ và một patch toàn ảnh (global patch là ảnh gốc).
    Đầu vào: [batch_size, num_patches, C, H, W] (num_patches = số patch cục bộ)
    Patch cuối cùng là ảnh gốc, không phải patch nhỏ.
    """

    def __init__(self, base_model, feature_dim, num_classes, num_patches):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        # +1 cho global patch (ảnh gốc)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * (num_patches + 1), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, num_patches, C, H, W] (patch cục bộ)
        batch_size, num_patches, C, H, W = x.size()
        # Xử lý patch cục bộ
        x_patches = x.view(-1, C, H, W)  # [batch_size * num_patches, C, H, W]
        feat_patches = self.base_model(
            x_patches
        )  # [batch_size * num_patches, feature_dim]
        feat_patches = feat_patches.view(batch_size, num_patches, self.feature_dim)

        # Tạo lại ảnh gốc từ các patch local có overlap
        # Giả sử chia dọc theo chiều cao, overlap mặc định là 0.2 (giống data.py)
        overlap_ratio = 0.2
        patch_height = H
        step = int(patch_height * (1 - overlap_ratio))
        full_height = step * (num_patches - 1) + patch_height
        full_img = torch.zeros(batch_size, C, full_height, W, device=x.device)
        count = torch.zeros(batch_size, 1, full_height, W, device=x.device)

        for i in range(num_patches):
            start_h = i * step
            end_h = start_h + patch_height
            full_img[:, :, start_h:end_h, :] += x[:, i]
            count[:, :, start_h:end_h, :] += 1

        # Trung bình cộng ở vùng overlap
        full_img = full_img / count.clamp(min=1.0)

        # Resize về kích thước patch (H, W)
        global_patch_resized = nn.functional.interpolate(
            full_img, size=(H, W), mode="bilinear", align_corners=False
        )  # [batch_size, C, H, W]
        feat_global = self.base_model(global_patch_resized)  # [batch_size, feature_dim]
        feat_global = feat_global.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Ghép lại: [batch_size, num_patches+1, feature_dim]
        feats = torch.cat([feat_patches, feat_global], dim=1)
        feats = feats + self.pos_embed
        feats = self.transformer_encoder(feats)
        feats = feats.contiguous().view(batch_size, -1)
        out = self.classifier(feats)
        return out


class PatchGlobalLocalTokenMixerClassifier(nn.Module):
    """
    Nhận đầu vào gồm các patch cục bộ và một patch toàn ảnh (global patch là ảnh gốc).
    Sử dụng token mixer (convolutional tokenizer + transformer encoder nhẹ) thay vì transformer encoder thuần.
    """

    def __init__(
        self, base_model, feature_dim, num_classes, num_patches, nhead=4, num_layers=2
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.reduced_dim = feature_dim // 4

        # Tokenizer cho từng patch (bao gồm global)
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, self.reduced_dim, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Positional embedding cho tất cả patch (local + global)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.reduced_dim))

        # Transformer encoder nhẹ cho token mixer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_dim,
            nhead=nhead,
            dim_feedforward=self.reduced_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim * (num_patches + 1), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, num_patches, C, H, W]
        batch_size, num_patches, C, H, W = x.size()
        # Local patch features
        x_patches = x.view(-1, C, H, W)
        feat_patches = self.base_model(
            x_patches
        )  # [batch_size * num_patches, feature_dim]
        if feat_patches.dim() == 2:
            feat_patches = feat_patches.view(
                batch_size * num_patches, self.feature_dim, 1, 1
            )
        # Tokenize local patches
        tokens_patches = self.tokenizer(
            feat_patches
        )  # [batch_size * num_patches, reduced_dim, 1, 1]
        tokens_patches = tokens_patches.view(batch_size, num_patches, self.reduced_dim)

        # Tạo lại ảnh gốc từ các patch local có overlap
        overlap_ratio = 0.2
        patch_height = H
        step = int(patch_height * (1 - overlap_ratio))
        full_height = step * (num_patches - 1) + patch_height
        full_img = torch.zeros(batch_size, C, full_height, W, device=x.device)
        count = torch.zeros(batch_size, 1, full_height, W, device=x.device)
        for i in range(num_patches):
            start_h = i * step
            end_h = start_h + patch_height
            full_img[:, :, start_h:end_h, :] += x[:, i]
            count[:, :, start_h:end_h, :] += 1
        full_img = full_img / count.clamp(min=1.0)
        # Resize về kích thước patch (H, W)
        global_patch_resized = nn.functional.interpolate(
            full_img, size=(H, W), mode="bilinear", align_corners=False
        )
        feat_global = self.base_model(global_patch_resized)  # [batch_size, feature_dim]
        if feat_global.dim() == 2:
            feat_global = feat_global.view(batch_size, self.feature_dim, 1, 1)
        tokens_global = self.tokenizer(feat_global)  # [batch_size, reduced_dim, 1, 1]
        tokens_global = tokens_global.view(batch_size, 1, self.reduced_dim)

        # Ghép lại: [batch_size, num_patches+1, reduced_dim]
        tokens = torch.cat([tokens_patches, tokens_global], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.contiguous().view(batch_size, -1)
        out = self.classifier(tokens)
        return out


class MILClassifier(nn.Module):
    """
    Globally-Aware Multiple Instance Learning Classifier
    Treats patches as instances in a bag and uses attention mechanism
    to aggregate patch features with global context awareness.
    Based on PatchGlobalLocalTokenMixerClassifier structure.
    """

    def __init__(
        self,
        base_model,
        feature_dim,
        num_classes,
        num_patches,
        attention_dim=128,
        nhead=4,
        num_layers=2,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.reduced_dim = feature_dim // 4

        # Tokenizer for patch feature reduction (similar to PatchGlobalLocalTokenMixerClassifier)
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, self.reduced_dim, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # MIL Attention mechanism for patch aggregation
        self.attention_V = nn.Sequential(
            nn.Linear(self.reduced_dim, attention_dim), nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.reduced_dim, attention_dim), nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(attention_dim, 1)

        # Global context feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(self.reduced_dim, self.reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Fusion layer for combining local MIL features with global features
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.reduced_dim + self.reduced_dim // 2, self.reduced_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.reduced_dim, self.reduced_dim // 2),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, num_patches, C, H, W]
        batch_size, num_patches, C, H, W = x.size()

        # Local patch features extraction (similar to PatchGlobalLocalTokenMixerClassifier)
        x_patches = x.view(-1, C, H, W)
        feat_patches = self.base_model(
            x_patches
        )  # [batch_size * num_patches, feature_dim]

        if feat_patches.dim() == 2:
            feat_patches = feat_patches.view(
                batch_size * num_patches, self.feature_dim, 1, 1
            )

        # Tokenize local patches
        tokens_patches = self.tokenizer(
            feat_patches
        )  # [batch_size * num_patches, reduced_dim, 1, 1]
        tokens_patches = tokens_patches.view(batch_size, num_patches, self.reduced_dim)

        # MIL Attention mechanism for patch aggregation
        A_V = self.attention_V(
            tokens_patches
        )  # [batch_size, num_patches, attention_dim]
        A_U = self.attention_U(
            tokens_patches
        )  # [batch_size, num_patches, attention_dim]
        A = self.attention_weights(A_V * A_U)  # [batch_size, num_patches, 1]
        A = torch.transpose(A, 2, 1)  # [batch_size, 1, num_patches]
        A = nn.functional.softmax(A, dim=2)  # Attention weights

        # Weighted aggregation of patch features (MIL aggregation)
        M = torch.bmm(A, tokens_patches)  # [batch_size, 1, reduced_dim]
        M = M.view(batch_size, -1)  # [batch_size, reduced_dim]

        # Global image reconstruction and feature extraction (from PatchGlobalLocalTokenMixerClassifier)
        overlap_ratio = 0.2
        patch_height = H
        step = int(patch_height * (1 - overlap_ratio))
        full_height = step * (num_patches - 1) + patch_height
        full_img = torch.zeros(batch_size, C, full_height, W, device=x.device)
        count = torch.zeros(batch_size, 1, full_height, W, device=x.device)

        for i in range(num_patches):
            start_h = i * step
            end_h = start_h + patch_height
            full_img[:, :, start_h:end_h, :] += x[:, i]
            count[:, :, start_h:end_h, :] += 1

        full_img = full_img / count.clamp(min=1.0)

        # Resize global image to patch size and extract features
        global_patch_resized = nn.functional.interpolate(
            full_img, size=(H, W), mode="bilinear", align_corners=False
        )
        feat_global = self.base_model(global_patch_resized)  # [batch_size, feature_dim]

        if feat_global.dim() == 2:
            feat_global = feat_global.view(batch_size, self.feature_dim, 1, 1)

        tokens_global = self.tokenizer(feat_global)  # [batch_size, reduced_dim, 1, 1]
        tokens_global = tokens_global.view(batch_size, self.reduced_dim)

        # Process global features
        processed_global = self.global_processor(
            tokens_global
        )  # [batch_size, reduced_dim//2]

        # Fuse local MIL features with global features (Globally-Aware)
        fused_features = torch.cat(
            [M, processed_global], dim=1
        )  # [batch_size, reduced_dim + reduced_dim//2]
        fused_features = self.fusion_layer(
            fused_features
        )  # [batch_size, reduced_dim//2]

        # Final classification
        logits = self.classifier(fused_features)

        return logits


class MILClassifierV2(nn.Module):
    """
    Multiple Instance Learning (MIL) Classifier - nâng cấp:
      - Gated attention pooling (CLAM-style)
      - Hỗ trợ mask (pad patch)
      - Head: LayerNorm + Dropout + Linear
      - Giữ API: output chỉ logits [B, num_classes], lưu attention trong self.last_attn_weights
    """

    def __init__(
        self,
        base_model,
        feature_dim,
        num_classes,
        attn_hidden=256,
        attn_dropout=0.1,
        head_dropout=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Gated attention
        self.attn_V = nn.Linear(feature_dim, attn_hidden)
        self.attn_U = nn.Linear(feature_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1)
        self.attn_drop = nn.Dropout(attn_dropout)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self._init_weights()

        # Nơi lưu attention để lấy sau khi forward
        self.last_attn_weights = None

    def _init_weights(self):
        for m in [self.attn_V, self.attn_U, self.attn_w]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _encode_patches(self, x):
        """
        x: [B, N, C, H, W] → feats: [B, N, D]
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.base_model(x)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, self.feature_dim)
        return feats

    def forward(self, x, mask=None, temperature=1.0):
        """
        x:    [B, N, C, H, W]
        mask: [B, N] (True = pad)
        return: logits [B, C]
        """
        feats = self._encode_patches(x)  # [B, N, D]

        # Gated attention
        V = torch.tanh(self.attn_V(feats))  # [B, N, H]
        U = torch.sigmoid(self.attn_U(feats))  # [B, N, H]
        scores = self.attn_w(self.attn_drop(V * U)).squeeze(-1)  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores / max(temperature, 1e-6), dim=1)  # [B, N]

        # Soft attention pooling
        pooled = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)  # [B, D]

        logits = self.head(pooled)  # [B, C]

        # Lưu lại attention để lấy sau
        self.last_attn_weights = attn_weights.detach()

        return logits


# ---------------- Gated Attention Pool (MIL) ----------------
class _GatedAttnPool(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(d_model, hidden)
        self.U = nn.Linear(d_model, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.constant_(self.V.bias, 0.0)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.constant_(self.U.bias, 0.0)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.constant_(self.w.bias, 0.0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, temperature: float = 1.0
    ):
        """
        x: [B, N, D]; mask: [B, N] (True = pad)
        """
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        scores = self.w(self.drop(v * u)).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores / max(temperature, 1e-6), dim=1)  # [B, N]
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, D]
        return pooled, attn


# ---------------- MIL_v3: Local + Global + Fusion ----------------
class MILClassifierV3(nn.Module):
    """
    MIL v3 (2-branch):
      - Local MIL over patch features (gated attention).
      - Global image reconstruction from patches + global backbone feature.
      - Fusion: 'concat' (default) or 'gated' (learnable mixing).
    Output:
      - logits [B, num_classes]
    Debug:
      - self.last_attn_weights  -> [B, N]
      - self.last_global_feat   -> [B, Dg]
      - self.last_local_feat    -> [B, Dl]
    """

    def __init__(
        self,
        base_model_local: nn.Module,  # CNN cho patch
        base_model_global: nn.Module,  # CNN cho ảnh global
        local_dim: int,  # D_local (output của base_model_local)
        global_dim: int,  # D_global (output của base_model_global)
        num_classes: int = 1,
        fusion: str = "concat",  # 'concat' | 'gated'
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        global_size: tuple = (448, 448),  # kích thước ảnh global sau reconstruct
        reconstruct_fn=None,  # callable(patches[B,N,C,h,w]) -> [B,C,H,W]; nếu None dùng vertical concat
        temperature: float = 1.0,
        overlap_ratio: float = 0.2,  # <— NEW
    ):
        super().__init__()
        assert fusion in ("concat", "gated"), "fusion must be 'concat' or 'gated'"
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes
        self.fusion = fusion
        self.global_size = global_size
        self.temperature = temperature
        self.reconstruct_fn = reconstruct_fn  # nếu None dùng _vertical_reconstruct

        # MIL pooling
        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        # Nếu fusion='concat' -> head nhận (Dl + Dg)
        # Nếu fusion='gated'  -> đưa Dg -> Dl và học gate để trộn (-> Dl)
        if fusion == "concat":
            fused_dim = local_dim + global_dim
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(head_dropout),
                nn.Linear(fused_dim, num_classes),
            )
        else:
            # Project global về local_dim rồi gated mix: y = g * local + (1-g) * proj(global)
            self.global_to_local = nn.Linear(global_dim, local_dim)
            self.gate = nn.Sequential(
                nn.Linear(
                    local_dim + local_dim, local_dim
                ),  # concat(local, gproj) -> hidden (DL)
                nn.ReLU(inplace=True),
                nn.Linear(local_dim, 1),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(local_dim),
                nn.Dropout(head_dropout),
                nn.Linear(local_dim, num_classes),
            )

        # dbg holders
        self.last_attn_weights = None
        self.last_global_feat = None
        self.last_local_feat = None

        # init head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        if fusion == "gated":
            nn.init.xavier_uniform_(self.global_to_local.weight)
            nn.init.constant_(self.global_to_local.bias, 0.0)

    # --------- Local encoder: từ patch ảnh -> [B,N,Dl] ---------
    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C, H, W] -> feats: [B, N, Dl]
        """
        B, N, C, H, W = x.shape
        x_ = x.view(B * N, C, H, W)
        feats = self.base_model_local(x_)
        if feats.dim() == 4:
            feats = (
                F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
            )  # [B*N, Dl?]
        feats = feats.view(B, N, -1)
        return feats

    # --------- Global reconstruct: từ patch -> ảnh global ---------
    def _vertical_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct ảnh theo chiều dọc có overlap (overlap-add).
        x: [B, N, C, h, w] (patch theo chiều cao, full width)
        """
        B, N, C, h, w = x.shape
        # step giống lúc cắt: step = patch_h * (1 - overlap_ratio)
        step = max(1, int(round(h * (1.0 - self.overlap_ratio))))
        # chiều cao canvas trước khi resize
        H_full = step * (N - 1) + h

        # canvas và weight để cộng chồng rồi chia
        canvas = x.new_zeros(B, C, H_full, w)
        weight = x.new_zeros(B, 1, H_full, w)

        # độ dài phần chồng lấn giữa 2 patch liên tiếp
        ovl = h - step  # có thể = 0 nếu overlap_ratio=0
        # vector trọng số theo chiều dọc (tam giác: 0→1 ở đầu, 1→0 ở cuối)
        base_w = torch.ones(h, device=x.device)
        if ovl > 0:
            ramp = torch.linspace(0.0, 1.0, steps=ovl, device=x.device)
            base_w[:ovl] = ramp  # phần đầu patch (đè lên đuôi patch trước)
            base_w[-ovl:] = 1.0 - ramp  # phần cuối patch (đè lên đầu patch sau)

        for i in range(N):
            y0 = i * step
            y1 = y0 + h

            # bản sao weight cho patch i, xử lý biên đầu/cuối
            w_i = base_w.clone()
            if ovl > 0:
                if i == 0:
                    w_i[:ovl] = 1.0  # không làm mờ ở mép trên ảnh
                if i == N - 1:
                    w_i[-ovl:] = 1.0  # không làm mờ ở mép dưới ảnh

            # mở rộng thành [1,1,h,1] để broadcast
            w_map = w_i.view(1, 1, h, 1)

            # cộng chồng có trọng số
            canvas[:, :, y0:y1, :] += x[:, i] * w_map
            weight[:, :, y0:y1, :] += w_map

        # tránh chia 0, rồi resize về global_size
        recon = canvas / weight.clamp(min=1e-6)
        recon = F.interpolate(
            recon, size=self.global_size, mode="bilinear", align_corners=False
        )
        return recon

    # --------- Global encoder: ảnh global -> [B, Dg] ---------
    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats_g = self.base_model_global(global_img)  # [B, Dg] or [B, Dg, h', w']
        if feats_g.dim() == 4:
            feats_g = F.adaptive_avg_pool2d(feats_g, 1).squeeze(-1).squeeze(-1)
        return feats_g  # [B, Dg]

    def forward(self, x_patches: torch.Tensor, mask: torch.Tensor = None):
        """
        x_patches: [B, N, C, H, W]
        mask     : [B, N] (True = pad/invalid patch)
        return   : logits [B, num_classes]
        """
        # ----- Local MIL -----
        feats_l = self._encode_patches(x_patches)  # [B, N, Dl]
        pooled_l, attn = self.mil_pool(
            feats_l, mask=mask, temperature=self.temperature
        )  # [B, Dl], [B, N]

        # ----- Global reconstruction + encoding -----
        if self.reconstruct_fn is not None:
            global_img = self.reconstruct_fn(x_patches)  # expect [B,C,H,W]
        else:
            global_img = self._vertical_reconstruct(x_patches)
        feats_g = self._encode_global(global_img)  # [B, Dg]

        # ----- Fusion -----
        if self.fusion == "concat":
            fused = torch.cat([pooled_l, feats_g], dim=1)  # [B, Dl+Dg]
        else:
            g_proj = self.global_to_local(feats_g)  # [B, Dl]
            gate = self.gate(torch.cat([pooled_l, g_proj], dim=1))  # [B,1]
            fused = gate * pooled_l + (1.0 - gate) * g_proj  # [B, Dl]

        logits = self.head(fused)

        # debug saves
        self.last_attn_weights = attn.detach()
        self.last_global_feat = feats_g.detach()
        self.last_local_feat = pooled_l.detach()

        return logits


def get_model(
    model_type="dinov2",
    num_classes=2,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
):
    num_patches = get_num_patches_from_config(config_path, num_patches)
    if model_type == "dinov2":
        model = DinoVisionTransformerClassifier(
            num_classes=num_classes, num_patches=num_patches
        )
    elif model_type in [
        "resnet50",
        "resnet101",
        "resnext50",
        "resnest50",
        "resnest50s2",
        "regnety",
        "fastervit",
        "convnextv2",
        "convnextv2_tiny",
        "efficientnetv2",
    ]:
        if model_type == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif model_type == "resnet101":
            base_model = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1
            )
            feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif model_type == "resnext50":
            base_model = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
            )
            feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif model_type == "resnest50":
            base_model = timm_models.create_model("resnest50d", pretrained=True)
            feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif model_type == "resnest50s2":
            base_model = timm_models.create_model("resnest50d_4s2x40d", pretrained=True)
            feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif model_type == "regnety":
            base_model = timm_models.create_model("regnety_080_tv", pretrained=True)
            feature_dim = base_model.head.fc.in_features
            base_model.head.fc = nn.Identity()
        elif model_type == "fastervit":
            try:
                base_model = create_model(
                    "faster_vit_0_any_res", pretrained=False, resolution=[448, 448]
                )
                feature_dim = base_model.head.in_features
                base_model.head = nn.Identity()
            except Exception as e:
                print(f"Warning: Failed to load 'faster_vit_0_any_res': {str(e)}")
                base_model = create_model(
                    "faster_vit_0_224", pretrained=True, checkpoint_path=None
                )
                feature_dim = base_model.head.in_features
                base_model.head = nn.Identity()
        elif model_type == "convnextv2":
            base_model = timm_models.create_model(
                "convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True
            )
            feature_dim = base_model.head.fc.in_features
            base_model.head.fc = nn.Identity()
        elif model_type == "convnextv2_tiny":
            base_model = timm_models.create_model(
                "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True
            )
            feature_dim = base_model.head.fc.in_features
            base_model.head.fc = nn.Identity()
        elif model_type == "efficientnetv2":
            base_model = timm_models.create_model("efficientnetv2_m", pretrained=False)
            feature_dim = base_model.classifier.in_features
            base_model.classifier = nn.Identity()

        if arch_type == "patch_resnet":
            model = PatchResNet(base_model, feature_dim, num_classes, num_patches)
        elif arch_type == "patch_transformer":
            model = PatchTransformerClassifier(
                base_model, feature_dim, num_classes, num_patches
            )
        elif arch_type == "token_mixer":
            model = TokenMixerClassifier(
                base_model, feature_dim, num_classes, num_patches, nhead=4, num_layers=2
            )
        elif arch_type == "global_local":
            model = PatchGlobalLocalClassifier(
                base_model, feature_dim, num_classes, num_patches
            )
        elif arch_type == "global_local_token":
            model = PatchGlobalLocalTokenMixerClassifier(
                base_model, feature_dim, num_classes, num_patches, nhead=4, num_layers=2
            )
        elif arch_type == "mil":
            model = MILClassifier(
                base_model,
                feature_dim,
                num_classes,
                num_patches,
                attention_dim=128,
                nhead=4,
                num_layers=2,
            )
        elif arch_type == "mil_v2":
            # Sử dụng MILClassifierV2 với các tham số mặc định
            model = MILClassifierV2(
                base_model=base_model,
                feature_dim=feature_dim,
                num_classes=num_classes,
                attn_hidden=256,
                attn_dropout=0.1,
                head_dropout=0.1,
            )
        elif arch_type == "mil_v3":
            # Ưu tiên tham số rõ ràng; nếu không truyền thì fallback từ feature_dim/base_model
            base_model_local = None
            base_model_global = None
            local_dim = None
            global_dim = None
            b_local = base_model_local if base_model_local is not None else base_model
            b_global = (
                base_model_global if base_model_global is not None else base_model
            )
            dl = local_dim if local_dim is not None else feature_dim
            dg = global_dim if global_dim is not None else feature_dim
            model = MILClassifierV3(
                base_model_local=b_local,
                base_model_global=b_global,
                local_dim=dl,
                global_dim=dg,
                num_classes=num_classes,
                attn_hidden=256,
                attn_dropout=0.1,
                head_dropout=0.1,
                global_size=(448, 448),  # kích thước ảnh global sau reconstruct
                reconstruct_fn=None,  # nếu cần có thể truyền hàm reconstruct riêng
                temperature=1.0,
                overlap_ratio=0.2,  # tỉ lệ overlap giữa các patch
            )
        else:
            raise ValueError(f"Unsupported arch_type: {arch_type}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model
