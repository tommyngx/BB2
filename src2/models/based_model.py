from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    get_dino_backbone,
    get_mamba_backbone,
)
from .head import get_linear_head
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def unfreeze_last_blocks(model, num_blocks=2):
    """
    Unfreeze the last `num_blocks` transformer blocks of a ViT/DINO backbone,
    or last blocks of ConvNeXt/DINOv3 (with stages attribute).
    """
    print(f"[INFO] Unfreezing last {num_blocks} blocks of the model...")

    # --- ViT-style (blocks/layers/transformer.blocks) ---
    block_attrs = ["blocks", "layers", "transformer.blocks"]
    for attr in block_attrs:
        blocks = None
        obj = model
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None
                break
        blocks = obj
        if blocks is not None and hasattr(blocks, "__getitem__"):
            total_blocks = len(blocks)
            unfrozen_layers = []
            for i in range(total_blocks - num_blocks, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                unfrozen_layers.append(i)
            # Freeze all other blocks
            for i in range(0, total_blocks - num_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = False
            print(f"[INFO] Unfroze {len(unfrozen_layers)} layers: {unfrozen_layers}")
            return

    # --- ConvNeXt/DINOv3-style (stages) ---
    if hasattr(model, "stages"):
        stages = model.stages
        for stage_idx, stage in enumerate(stages):
            if hasattr(stage, "blocks"):
                blocks = stage.blocks
                total_blocks = len(blocks)
                unfrozen_layers = []
                for i in range(total_blocks - num_blocks, total_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = True
                    unfrozen_layers.append(i)
                for i in range(0, total_blocks - num_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = False
                print(
                    f"[INFO] Stage {stage_idx}: Unfroze {len(unfrozen_layers)} blocks: {unfrozen_layers}"
                )
            else:
                print(f"[WARN] Stage {stage_idx} has no 'blocks' attribute.")
        if hasattr(model, "stem"):
            for module in model.stem.modules():
                for param in getattr(module, "parameters", lambda: [])():
                    param.requires_grad = True
            print("[INFO] Unfroze stem layers.")
        return

    print("[WARN] No recognized block structure found for unfreezing.")


def get_based_model(model_type="resnet50", num_classes=2, dino_unfreeze_blocks=2):
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        # Replace the head with a linear classifier
        model = backbone
        model.fc = get_linear_head(feature_dim, num_classes)
    elif model_type in ["mamba_t", "mamba_s"]:
        model, feature_dim = get_mamba_backbone(model_type, num_classes=num_classes)
        # model.model.head = nn.Linear(feature_dim, num_classes)
        # print("Using Mamba_T backbone with custom head.", model)
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
        "maxvit_base",  # thêm maxvit_base
        "eva02_small",  # sửa lại đúng tên model_type
        "eva02_base",  # thêm eva02_base
        "vit_small",  # thêm model mới
        "swinv2_tiny",
        "swinv2_base",
        "swinv2_small",
        "mambaout_tiny",
    ]:
        model, feature_dim = get_timm_backbone(model_type)
        # Replace the head with a linear classifier for all timm backbones
        if hasattr(model, "fc"):
            model.fc = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "head") and hasattr(model.head, "fc"):
            model.head.fc = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "head"):
            model.head = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "classifier"):
            model.classifier = get_linear_head(feature_dim, num_classes)
        else:
            raise ValueError("Unknown head structure for timm backbone")
        # elif model_type == "fastervit":
        #    model, feature_dim = get_fastervit_backbone()
        # model.head = get_linear_head(feature_dim, num_classes)
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
        transformer, feature_dim = get_dino_backbone(model_type)

        model = DinoVisionTransformerClassifier(transformer, feature_dim, num_classes)

        # DEBUG: In ra class của transformer để kiểm tra
        print(f"[DEBUG] DINO backbone type: {type(transformer)}")
        # Freeze toàn bộ backbone trước
        for param in model.transformer.parameters():
            param.requires_grad = False
        # Unfreeze last blocks cho đúng backbone gốc
        unfreeze_last_blocks(model.transformer, dino_unfreeze_blocks)
        # Đảm bảo head classifier luôn trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unsupported model_type for base model")
    return model


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        transformer,
        feature_dim,
        num_classes,
        hidden_dim=512,
        dropout_p=0.3,
    ):
        super().__init__()
        self.transformer = transformer
        self.feature_dim = feature_dim
        # self.classifier = get_linear_head(feature_dim, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x
