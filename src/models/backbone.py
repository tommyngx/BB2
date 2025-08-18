import torch.nn as nn
import torchvision.models as models
import timm.models as timm_models

# from fastervit import create_model
import torch.hub as hub
import torch.serialization
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def get_resnet_backbone(model_type="resnet50"):
    if model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnext50":
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError("Unsupported resnet backbone type")
    return model, feature_dim


def get_timm_backbone(model_type):
    if model_type == "resnest50":
        model = timm_models.create_model("resnest50d", pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnest101":
        model = timm_models.create_model("resnest101e", pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnest50s2":
        model = timm_models.create_model("resnest50d_4s2x40d", pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "regnety":
        model = timm_models.create_model("regnety_080_tv", pretrained=True)
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "convnextv2":
        model = timm_models.create_model(
            "convnextv2_small.fcmae_ft_in22k_in1k", pretrained=True
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "convnextv2_tiny":
        model = timm_models.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "efficientnetv2":
        model = timm_models.create_model("efficientnetv2_m", pretrained=False)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError("Unsupported timm backbone type")
    return model, feature_dim


def get_fastervit_backbone():
    try:
        model = create_model(
            "faster_vit_0_any_res", pretrained=False, resolution=[448, 448]
        )
        feature_dim = model.head.in_features
        model.head = nn.Identity()
    except Exception:
        model = create_model("faster_vit_0_224", pretrained=True, checkpoint_path=None)
        feature_dim = model.head.in_features
        model.head = nn.Identity()
    return model, feature_dim


def get_dino_backbone(
    model_type="dinov2_vitb14", weights=None, repo_dir="facebookresearch/dinov3"
):
    """
    Supported model_type:
        - dinov2_vitb14 (default)
        - dinov2_vits14
        - dinov3_vits16
        - dinov3_vits16plus
        - dinov3_vitb16
        - dinov3_convnext_tiny
        - dinov3_convnext_small
    """
    if model_type == "dinov2_vitb14":
        transformer = hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        feature_dim = transformer.norm.normalized_shape[0]
        return transformer, feature_dim
    elif model_type == "dinov2_vits14":
        transformer = hub.load("facebookresearch/dinov2", "dinov2_vits14")
        feature_dim = transformer.norm.normalized_shape[0]
        return transformer, feature_dim
    elif model_type == "dinov3_vits16":
        transformer = hub.load(
            repo_dir, "dinov3_vits16", source="local", weights=weights
        )
        feature_dim = transformer.norm.normalized_shape[0]
        # Để freeze backbone khi train:
        # for param in transformer.parameters():
        #     param.requires_grad = False
        return transformer, feature_dim
    elif model_type == "dinov3_vits16plus":
        transformer = hub.load(
            repo_dir, "dinov3_vits16plus", source="local", weights=weights
        )
        feature_dim = transformer.norm.normalized_shape[0]
        # Để freeze backbone khi train:
        # for param in transformer.parameters():
        #     param.requires_grad = False
        return transformer, feature_dim
    elif model_type == "dinov3_vitb16":
        transformer = hub.load(
            repo_dir, "dinov3_vitb16", source="local", weights=weights
        )
        feature_dim = transformer.norm.normalized_shape[0]
        # Để freeze backbone khi train:
        # for param in transformer.parameters():
        #     param.requires_grad = False
        return transformer, feature_dim
    elif model_type == "dinov3_convnext_tiny":
        transformer = hub.load(
            repo_dir, "dinov3_convnext_tiny", source="local", weights=weights
        )
        feature_dim = (
            transformer.norm.normalized_shape[0]
            if hasattr(transformer, "norm")
            else transformer.head.in_features
        )
        # Để freeze backbone khi train:
        # for param in transformer.parameters():
        #     param.requires_grad = False
        return transformer, feature_dim
    elif model_type == "dinov3_convnext_small":
        transformer = hub.load(
            repo_dir, "dinov3_convnext_small", source="local", weights=weights
        )
        feature_dim = (
            transformer.norm.normalized_shape[0]
            if hasattr(transformer, "norm")
            else transformer.head.in_features
        )
        # Để freeze backbone khi train:
        # for param in transformer.parameters():
        #     param.requires_grad = False
        return transformer, feature_dim
    else:
        raise ValueError("Unsupported DINO backbone type")
