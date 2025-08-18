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


def get_dino_backbone(model_type="dinov2_vitb14", weights=None):
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
    dino2_models = {
        "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
        "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
    }
    # HuggingFace model hub ids for dinov3
    dino3_models = {
        # "dinov3_vits16": "facebook/dinov3-vits16",
        "dinov3_vits16": "Fanqi-Lin-IR/dinov3_vits16_pretrain",
        # "dinov3_vits16plus": "facebook/dinov3-vits16plus",
        "dinov3_vith16plus": "Fanqi-Lin-IR/dinov3_vith16plus",
        "dinov3_vitb16": "facebook/dinov3-vitb16",
        "dinov3_convnext_tiny": "facebook/dinov3-convnext-tiny",
        "dinov3_convnext_small": "facebook/dinov3-convnext-small",
    }

    if model_type in dino2_models:
        repo, name = dino2_models[model_type]
        transformer = hub.load(repo, name)
        # Common way to get feature_dim for all DINOv2 models
        if hasattr(transformer, "norm") and hasattr(
            transformer.norm, "normalized_shape"
        ):
            feature_dim = transformer.norm.normalized_shape[0]
        else:
            raise RuntimeError("Cannot determine feature_dim for this DINOv2 backbone")
        return transformer, feature_dim
    elif model_type == "dinov3_vits16":
        from huggingface_hub import hf_hub_download
        import torch
        import timm

        local_path = hf_hub_download(
            repo_id="Fanqi-Lin-IR/dinov3_vits16_pretrain",
            filename="dinov3_vits16_pretrain.pth",
        )
        # Load ViT-S/16 architecture from timm with correct img_size
        model = timm.create_model("vit_small_patch16_224", pretrained=False)
        # Set model's img_size to 448 if you want to use 448x448 input
        model.patch_embed.img_size = (448, 448)
        model.img_size = (448, 448)
        # Update model.patch_embed.num_patches and model.pos_embed accordingly
        model.patch_embed.num_patches = (448 // model.patch_embed.patch_size[0]) * (
            448 // model.patch_embed.patch_size[1]
        )
        # Recreate pos_embed with correct shape
        import math

        num_patches = model.patch_embed.num_patches
        embed_dim = model.embed_dim
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        torch.nn.init.trunc_normal_(model.pos_embed, std=0.02)

        state_dict = torch.load(local_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Fix positional embedding shape mismatch if needed
        if "pos_embed" in state_dict:
            pos_embed_pretrained = state_dict["pos_embed"]
            pos_embed_model = model.pos_embed
            if pos_embed_pretrained.shape != pos_embed_model.shape:
                # Interpolate positional embedding
                cls_token = pos_embed_pretrained[:, 0:1, :]
                pos_tokens = pos_embed_pretrained[:, 1:, :]
                old_grid_size = int(pos_tokens.shape[1] ** 0.5)
                new_grid_size = int(num_patches**0.5)
                pos_tokens = pos_tokens.reshape(
                    1, old_grid_size, old_grid_size, -1
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_grid_size, new_grid_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    1, new_grid_size * new_grid_size, -1
                )
                new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
                state_dict["pos_embed"] = new_pos_embed
        model.load_state_dict(state_dict, strict=False)
        feature_dim = model.head.in_features
        model.head = nn.Identity()
        return model, feature_dim
    elif model_type in [
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
    ]:
        model_id = dino3_models[model_type]
        from transformers import AutoImageProcessor, AutoModel

        # processor = AutoImageProcessor.from_pretrained(model_id)
        transformer = AutoModel.from_pretrained(model_id)
        # Try to get feature_dim from transformer config or attributes
        if hasattr(transformer, "config") and hasattr(
            transformer.config, "hidden_size"
        ):
            feature_dim = transformer.config.hidden_size
        elif hasattr(transformer, "norm") and hasattr(
            transformer.norm, "normalized_shape"
        ):
            feature_dim = transformer.norm.normalized_shape[0]
        elif hasattr(transformer, "head") and hasattr(transformer.head, "in_features"):
            feature_dim = transformer.head.in_features
        else:
            raise RuntimeError("Cannot determine feature_dim for this DINOv3 backbone")
        # Return processor if needed for preprocessing, else just transformer and feature_dim
        return transformer, feature_dim
    else:
        raise ValueError("Unsupported DINO backbone type")
