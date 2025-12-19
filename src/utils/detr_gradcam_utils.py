"""
GradCAM utilities for DETR models
- Layer detection for different backbones
"""


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

    print(f"🔍 Detecting GradCAM layer for model: {model_name}")
    print(f"   Has backbone: {has_backbone}")

    def find_deepest_layer(model, candidates):
        """Find the deepest valid layer from candidates"""
        named_modules = dict([*model.named_modules()])
        print(f"   Searching among {len(named_modules)} modules...")
        for cand in candidates:
            for name in named_modules:
                if name.endswith(cand):
                    print(f"   ✓ Found matching layer: {name}")
                    return name
        print(f"   ✗ No matching layer found from candidates: {candidates}")
        return None

    # ResNet, ResNeXt, ResNeSt
    if "resnet" in model_name or "resnext" in model_name or "resnest" in model_name:
        layer = f"{prefix}layer4"
        print(f"   → Selected layer: {layer}")
        return layer

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

    # DINOv2
    elif "dinov2" in model_name or "dinov3" in model_name:
        layer = f"{prefix}transformer.blocks.23.norm1"
        print(f"   → Selected layer: {layer}")
        return layer

    # Swin Transformer
    elif "swin" in model_name:
        layer = f"{prefix}layers.-1.blocks.-1.norm1"
        print(f"   → Selected layer: {layer}")
        return layer

    # EVA02
    elif "eva02" in model_name:
        layer = f"{prefix}blocks.-1.norm1"
        print(f"   → Selected layer: {layer}")
        return layer

    # Fallback: try to get last layer
    else:
        print(f"   ⚠️ Unknown model type, using fallback...")
        children = list(model.named_children())
        if children:
            last_child_name = children[-1][0]
            if last_child_name == "backbone" and has_backbone:
                backbone = children[-1][1]
                backbone_children = list(backbone.named_children())
                if backbone_children:
                    layer = f"backbone.{backbone_children[-1][0]}"
                    print(f"   → Fallback layer: {layer}")
                    return layer
            print(f"   → Fallback layer: {last_child_name}")
            return last_child_name
        print(f"   ✗ No suitable layer found!")
        return None
