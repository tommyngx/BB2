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
        self, base_model, feature_dim, num_classes, num_patches, nhead=8, num_layers=2
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        # Learnable positional encoding for each patch
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim))
        # Transformer encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, batch_first=True
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
        x = x.view(-1, C, H, W)  # [batch_size * num_patches, C, H, W]
        x = self.base_model(x)  # [batch_size * num_patches, feature_dim]
        x = x.view(
            batch_size, num_patches, self.feature_dim
        )  # [batch_size, num_patches, feature_dim]
        x = x + self.pos_embed  # Add positional encoding
        x = self.transformer_encoder(x)  # [batch_size, num_patches, feature_dim]
        x = x.contiguous().view(
            batch_size, -1
        )  # [batch_size, num_patches * feature_dim]
        x = self.classifier(x)
        return x


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
        else:
            raise ValueError(f"Unsupported arch_type: {arch_type}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model
