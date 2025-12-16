import torch
import torch.nn as nn
from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    get_dino_backbone,
    get_mamba_backbone,
)


class SpatialAttention(nn.Module):
    """Spatial attention on feature map"""

    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        attn_map = self.attention(x)  # [B, 1, H, W]
        return x * attn_map, attn_map


class M2Model(nn.Module):
    """Multi-task model: classification + bbox regression"""

    def __init__(self, backbone, feature_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Spatial attention for bbox
        self.spatial_attention = SpatialAttention(feature_dim)

        # BBox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # normalized [0,1]
        )

    def forward(self, x):
        # Backbone feature map [B, C, H, W]
        feat_map = self.backbone(x)

        # Classification branch
        cls_feat = self.global_pool(feat_map).flatten(1)
        cls_output = self.classifier(cls_feat)

        # BBox branch with attention
        attn_feat, attn_map = self.spatial_attention(feat_map)
        bbox_feat = self.global_pool(attn_feat).flatten(1)
        bbox_output = self.bbox_head(bbox_feat)

        return cls_output, bbox_output, attn_map


def get_m2_model(model_type="resnet50", num_classes=2):
    """Get multi-task model based on model_type"""
    # Get backbone
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.Identity()  # Keep feature map

    elif model_type in ["mamba_t", "mamba_s"]:
        raise ValueError(
            f"Mamba models don't support spatial feature maps for M2. Use CNN-based models."
        )

    elif model_type in [
        "resnest50",
        "resnest101",
        "resnest50s2",
        "regnety",
        "convnextv2base",
        "convnextv2_tiny",
        "efficientnetv2",
        "efficientnetv2s",
        "maxvit_tiny",
        "maxvit_small",
        "maxvit_base",
        "eva02_small",
        "eva02_base",
        "vit_small",
        "swinv2_tiny",
        "swinv2_base",
        "swinv2_small",
        "mambaout_tiny",
    ]:
        backbone, feature_dim = get_timm_backbone(model_type)

        # Wrap to get feature map
        if hasattr(backbone, "forward_features"):

            class TimmFeatureWrapper(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model

                def forward(self, x):
                    feat = self.base_model.forward_features(x)
                    # Handle different output formats
                    if feat.ndim == 4 and feat.shape[-1] < feat.shape[1]:
                        return feat  # Already [B, C, H, W]
                    elif feat.ndim == 4:
                        return feat.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    elif feat.ndim == 3:
                        # ViT: [B, N, C] -> [B, C, H, W]
                        B, N, C = feat.shape
                        H = W = int(N**0.5)
                        return feat.transpose(1, 2).reshape(B, C, H, W)
                    return feat

            backbone = TimmFeatureWrapper(backbone)
        else:
            # Fallback: remove heads
            if hasattr(backbone, "fc"):
                backbone.fc = nn.Identity()
            elif hasattr(backbone, "head") and hasattr(backbone.head, "fc"):
                backbone.head.fc = nn.Identity()
            elif hasattr(backbone, "head"):
                backbone.head = nn.Identity()
            elif hasattr(backbone, "classifier"):
                backbone.classifier = nn.Identity()

    elif model_type in [
        "dinov2_small",
        "dinov2_base",
        "dinov2_small_reg",
        "dinov2_base_reg",
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_vit16small",
        "dinov3_vit16smallplus",
        "dinov3_vit16base",
        "dinov3_vit16large",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
        "medino_vitb16",
        "dinov2uni_base",
    ]:
        backbone, feature_dim = get_dino_backbone(model_type)

        # Wrap Dino/ViT to get spatial features
        class DinoFeatureWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                if hasattr(self.base_model, "forward_features"):
                    feat = self.base_model.forward_features(x)
                else:
                    feat = self.base_model(x)

                # [B, N+1, C] -> [B, C, H, W] (remove CLS token)
                if feat.ndim == 3:
                    B, N, C = feat.shape
                    feat = feat[:, 1:, :]  # Remove CLS
                    H = W = int((N - 1) ** 0.5)
                    return feat.transpose(1, 2).reshape(B, C, H, W)
                return feat

        backbone = DinoFeatureWrapper(backbone)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Create M2 model
    model = M2Model(backbone, feature_dim, num_classes)
    return model
