import torch
import torch.nn as nn
from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    get_dino_backbone,
    get_mamba_backbone,
)


class SpatialAttention(nn.Module):
    """Spatial attention for bbox regression"""

    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, feature_dim]
        attention_weights = self.attention(x)  # [B, 1]
        return x * attention_weights


class M2Model(nn.Module):
    """
    Multi-task model for classification + bbox regression
    """

    def __init__(self, backbone, feature_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Classification head
        self.classifier = nn.Sequential(
            # nn.Linear(feature_dim, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(512, num_classes),
            nn.Linear(feature_dim, num_classes)
        )

        # Bbox regression head with attention
        self.spatial_attention = SpatialAttention(feature_dim)

        self.bbox_regressor = nn.Sequential(
            # nn.Linear(feature_dim, 512),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # Output normalized coords [0, 1]
        )

    def forward(self, x):
        # Get backbone features
        features = self.backbone(x)

        # Classification output
        cls_output = self.classifier(features)

        # Apply spatial attention for bbox
        attended_features = self.spatial_attention(features)

        # Bbox regression output
        bbox_output = self.bbox_regressor(attended_features)

        return cls_output, bbox_output


def get_m2_model(model_type="resnet50", num_classes=2):
    """
    Get multi-task model based on model_type
    """
    # Get backbone
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        # Remove fc layer
        backbone.fc = nn.Identity()
    elif model_type in ["mamba_t", "mamba_s"]:
        backbone, feature_dim = get_mamba_backbone(model_type, num_classes=None)
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
        # Remove head
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
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Create multi-task model
    model = M2Model(backbone, feature_dim, num_classes)

    return model
