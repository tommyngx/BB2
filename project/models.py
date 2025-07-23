import torch.nn as nn
import torchvision.models as models
import timm
from fastervit import create_model
import torch.hub as hub
import torch.serialization
import argparse
import warnings

torch.serialization.add_safe_globals([argparse.Namespace])
warnings.filterwarnings("ignore", category=FutureWarning)


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.norm.normalized_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


def get_model(model_type="dinov2", num_classes=2):
    if model_type == "dinov2":
        model = DinoVisionTransformerClassifier(num_classes=num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnext50":
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "fastervit":
        try:
            model = create_model(
                "faster_vit_0_any_res", pretrained=False, resolution=[448, 448]
            )
            state_dict = model.state_dict()
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            print(
                f"⚠️ Ignored {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys when loading FasterViT."
            )
            model.head = nn.Linear(model.head.in_features, num_classes)
        except Exception as e:
            print(
                f"Warning: Failed to load 'faster_vit_0_any_res' with weights_only=True: {str(e)}"
            )
            print(
                "Falling back to 'faster_vit_0_224' with weights_only=False. Ensure the checkpoint is from a trusted source."
            )
            model = create_model(
                "faster_vit_0_224",
                pretrained=True,
                checkpoint_path=None,
                weights_only=False,
            )
            model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_type == "convnextv2":
        model = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
    elif model_type == "efficientnetv2":
        model = timm.create_model(
            "efficientnetv2_m", pretrained=False, num_classes=num_classes
        )
    else:
        raise ValueError(
            "model_type must be 'dinov2', 'resnet50', 'resnet101', 'resnext50', 'fastervit', 'convnextv2', or 'efficientnetv2'"
        )
    return model
