"""
GradCAM utilities for DETR models
- Layer detection for different backbones
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from numpy import ndarray


def vit_reshape_transform(x, grid_h, grid_w):
    """
    Reshape ViT output từ [B, N, C] sang [B, C, H, W]
    Loại bỏ CLS token và extra tokens
    """
    B, N, C = x.shape
    num_patches = grid_h * grid_w

    # Tính số extra tokens (CLS + register tokens)
    num_extra = N - num_patches

    if num_extra < 0:
        raise ValueError(
            f"Mismatch: Expected {num_patches} patches but got {N} tokens. "
            f"Grid size ({grid_h}, {grid_w}) may be incorrect."
        )

    # Loại bỏ CLS token và register tokens (thường ở đầu)
    x = x[:, num_extra:, :]  # [B, num_patches, C]

    if x.shape[1] != num_patches:
        raise ValueError(
            f"After removing {num_extra} extra tokens, got {x.shape[1]} tokens "
            f"but expected {num_patches} patches."
        )

    # Reshape thành spatial grid
    x = x.reshape(B, grid_h, grid_w, C)  # [B, H, W, C]
    return x.permute(0, 3, 1, 2)  # [B, C, H, W]


def gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str | nn.Module,
    class_idx: int | None = None,
) -> ndarray:
    """
    GradCAM for both CNN and ViT. Auto-detects model type and reshapes accordingly.
    """
    # Giữ model ở eval mode nhưng enable gradient
    model.eval()

    # Enable gradient cho tất cả parameters
    for param in model.parameters():
        param.requires_grad = True

    activations = []
    gradients = []

    # Cho phép truyền tên lớp hoặc module
    if isinstance(target_layer, str):
        try:
            layer = dict([*model.named_modules()])[target_layer]
        except KeyError:
            raise ValueError(
                f"Layer '{target_layer}' not found in model. "
                f"Available layers: {list(dict([*model.named_modules()]).keys())}"
            )
    else:
        layer = target_layer

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    # Enable gradient computation cho forward pass
    with torch.set_grad_enabled(True):
        output = model(input_tensor)

        # ADDED: Handle DETR model output (returns dict instead of tensor)
        if isinstance(output, dict):
            # DETR models return dict with 'cls_logits' key
            if "cls_logits" in output:
                output = output[
                    "cls_logits"
                ]  # Extract classification logits [B, num_classes]
            else:
                raise ValueError(
                    f"Model returned dict but no 'cls_logits' key found. "
                    f"Available keys: {list(output.keys())}"
                )

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, class_idx].backward()

    # Kiểm tra hooks có capture được không
    if len(activations) == 0 or len(gradients) == 0:
        handle_f.remove()
        handle_b.remove()
        raise RuntimeError(
            f"Hooks failed for layer '{target_layer}'. "
            f"Activations: {len(activations)}, Gradients: {len(gradients)}\n"
            f"This may happen if the model is frozen or layer doesn't participate in backward pass."
        )

    acts = activations[0]
    grads = gradients[0]

    print(f"[DEBUG] acts shape: {acts.shape}, grads shape: {grads.shape}")

    # ===== Kiểm tra cấu trúc wrapper và lấy grid_size =====
    is_vit = False
    grid_h, grid_w = None, None

    # **TÍNH GRID_SIZE TỪ INPUT TENSOR THỰC TẾ**
    actual_input_h, actual_input_w = input_tensor.shape[2], input_tensor.shape[3]

    # Kiểm tra DinoVisionTransformerClassifier wrapper
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "patch_embed"):
            patch_size = None

            # Lấy patch_size
            if hasattr(model.transformer.patch_embed, "patch_size"):
                patch_size = (
                    model.transformer.patch_embed.patch_size[0]
                    if hasattr(model.transformer.patch_embed.patch_size, "__getitem__")
                    else model.transformer.patch_embed.patch_size
                )

            if patch_size is not None:
                is_vit = True
                # **TÍNH TỪ INPUT THỰC TẾ, KHÔNG DÙNG model.patch_embed.img_size**
                grid_h = actual_input_h // patch_size
                grid_w = actual_input_w // patch_size
            else:
                # Fallback
                if hasattr(model.transformer.patch_embed, "grid_size"):
                    is_vit = True
                    grid_h, grid_w = model.transformer.patch_embed.grid_size
                elif hasattr(model.transformer.patch_embed, "num_patches"):
                    is_vit = True
                    num_patches = model.transformer.patch_embed.num_patches
                    grid_h = grid_w = int(num_patches**0.5)
                    # print(
                    #     f"DEBUG: Calculated from num_patches, grid_size=({grid_h}, {grid_w})"
                    # )

    # Kiểm tra direct ViT model
    elif hasattr(model, "patch_embed"):
        patch_size = None

        if hasattr(model.patch_embed, "patch_size"):
            patch_size = (
                model.patch_embed.patch_size[0]
                if hasattr(model.patch_embed.patch_size, "__getitem__")
                else model.patch_embed.patch_size
            )

        if patch_size is not None:
            is_vit = True
            # **TÍNH TỪ INPUT THỰC TẾ**
            grid_h = actual_input_h // patch_size
            grid_w = actual_input_w // patch_size
        else:
            # Fallback
            if hasattr(model.patch_embed, "grid_size"):
                is_vit = True
                grid_h, grid_w = model.patch_embed.grid_size
            elif hasattr(model.patch_embed, "num_patches"):
                is_vit = True
                num_patches = model.patch_embed.num_patches
                grid_h = grid_w = int(num_patches**0.5)
                # print(
                #     f"DEBUG: Calculated from num_patches, grid_size=({grid_h}, {grid_w})"
                # )

    # Reshape nếu là ViT và activation có dạng [B, N, C]
    if is_vit and acts.ndim == 3 and grid_h is not None and grid_w is not None:
        try:
            acts = vit_reshape_transform(acts, grid_h, grid_w)  # [B, C, H, W]
            grads = vit_reshape_transform(grads, grid_h, grid_w)  # [B, C, H, W]
        except ValueError as e:
            is_vit = False
    elif is_vit and acts.ndim == 3:
        pass

    # GradCAM logic
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.uint8(cam * 255)

    handle_f.remove()
    handle_b.remove()

    return cam


def get_gradcam_layer(model, model_name):
    """
    Return the appropriate layer name for GradCAM based on model_name.
    Automatically detects if model has backbone wrapper and adds prefix.

    Args:
        model: The model instance
        model_name: Model architecture name (e.g., 'resnet50', 'dinov2_small')

    Returns:
        str: Layer name for GradCAM (e.g., 'backbone.layer4' for DETR models)
    """
    # Check if model has backbone (DETR models)
    has_backbone = hasattr(model, "backbone")
    prefix = "backbone." if has_backbone else ""

    def find_deepest_layer(model, candidates):
        """Find the deepest valid layer from candidates"""
        named_modules = dict([*model.named_modules()])
        for cand in candidates:
            for name in named_modules:
                if name.endswith(cand):
                    return name
        return None

    # ResNet, ResNeXt, ResNeSt
    if "resnet" in model_name or "resnext" in model_name or "resnest" in model_name:
        return f"{prefix}layer4"

    # ConvNeXt
    elif "convnext" in model_name:
        candidates = [
            "base_model.stages.3.blocks.2.conv_dw",
            "stages.3.blocks.2.conv_dw",
            "stages.3.blocks.2",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # MaxViT
    elif "maxvit" in model_name:
        candidates = [
            "base_model.stages.3.blocks.1.conv.norm2",
            "base_model.stages.3.blocks.1.conv",
            "base_model.stages.3.blocks.1",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # RegNetY
    elif "regnety" in model_name:
        candidates = [
            "base_model.s4.b1.conv3.conv",
            "s4.b1.conv3.conv",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # EfficientNet
    elif "efficientnet" in model_name:
        candidates = [
            "base_model.blocks.5.2.conv_pwl",
            "base_model.blocks.5.2",
            "blocks.5.2.conv_pwl",
            "blocks.5.2",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # DINOv2 and DINOv3
    # elif "dinov2" in model_name or "dinov3" in model_name:
    #    return f"{prefix}transformer.blocks.23.norm1"

    elif "dinov2" in model_name or "dinov3" in model_name or "vit" in model_name:
        # Tìm block cuối cùng của backbone.base_model.blocks
        named_modules = dict([*model.named_modules()])
        block_names = [
            k for k in named_modules if k.endswith("norm1") and "blocks." in k
        ]
        if block_names:
            # Lấy block.norm1 có số lớn nhất
            last_block = sorted(
                block_names, key=lambda x: int(x.split("blocks.")[1].split(".")[0])
            )[-1]
            return last_block
        # Fallback
        return f"{prefix}base_model.blocks.0.norm1"

    # Swin Transformer
    elif "swin" in model_name:
        return f"{prefix}layers.-1.blocks.-1.norm1"

    # EVA02
    elif "eva02" in model_name:
        return f"{prefix}blocks.-1.norm1"

    # Fallback: try to get last layer
    else:
        children = list(model.named_children())
        if children:
            last_child_name = children[-1][0]
            if last_child_name == "backbone" and has_backbone:
                backbone = children[-1][1]
                backbone_children = list(backbone.named_children())
                if backbone_children:
                    return f"backbone.{backbone_children[-1][0]}"
            return last_child_name
        return None
