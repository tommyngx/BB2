import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import cv2
import torch
from torchvision import transforms
import random


def get_image_size_from_config(config_path="config/config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    img_size = config.get("image_size", 448)
    if isinstance(img_size, (list, tuple)):
        return tuple(img_size)
    return (img_size, img_size)


def get_target_column_from_config(config_path="config/config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("target_column", "cancer")


def get_num_patches_from_config(config_path="config/config.yaml", num_patches=None):
    if num_patches is not None:
        return num_patches
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("num_patches", 2)  # Default 2 patches


def get_model_key(
    config_path="config/config.yaml", base_model_key="dataset", num_patches=None
):
    num_patches = get_num_patches_from_config(config_path, num_patches)
    patch_suffix = f"p{num_patches}"
    return f"{base_model_key}_{patch_suffix}"


def load_data(data_folder, config_path="config/config.yaml"):
    target_column = get_target_column_from_config(config_path)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df = df.drop_duplicates(subset=["link"])
    if not np.issubdtype(df[target_column].dtype, np.number):
        df["target_label"], class_names = pd.factorize(df[target_column])
        print(
            f"Detected non-numeric target labels in column '{target_column}'. Mapping:"
        )
        for idx, name in enumerate(class_names):
            print(f"  {idx}: {name}")
        label_col = "target_label"
    else:
        label_col = target_column
        class_names = sorted(df[target_column].unique())
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    print("Train set class distribution:")
    print(train_df[label_col].value_counts())
    print("Test set class distribution:")
    print(test_df[label_col].value_counts())
    if train_df[label_col].nunique() > 2:
        print("Detected multi-class classification. Classes:", list(class_names))
    train_df["cancer"] = train_df[label_col]
    test_df["cancer"] = test_df[label_col]
    return train_df, test_df


def split_image_into_patches(image, num_patches=2):
    """Split the original image into num_patches along the height with equal sizes."""
    image = np.array(image)
    height, width = image.shape[:2]
    patch_height = height // num_patches
    patches = []
    for i in range(num_patches):
        start_h = i * patch_height
        end_h = (i + 1) * patch_height
        if i == num_patches - 1:
            end_h = height
        patch = image[start_h:end_h, :, :]
        if patch.shape[0] < patch_height:
            pad_height = patch_height - patch.shape[0]
            patch = np.pad(
                patch,
                ((0, pad_height), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        patches.append(patch)
    return patches


class CancerPatchDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, num_patches=2):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.num_patches = num_patches
        self.img_size = get_image_size_from_config()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])

        # Apply augmentation to the whole image
        if self.transform:
            image_np = np.array(image)
            transform_with_resize = A.Compose(
                [A.Resize(*self.img_size), *self.transform]
            )
            augmented = transform_with_resize(image=image_np)
            image = augmented["image"]
        else:
            image = np.array(image)
            image = cv2.resize(image, self.img_size)
            image = A.Normalize([0.5] * 3, [0.5] * 3)(image=image)["image"]
            image = ToTensorV2()(image=image)["image"]

        # Split the augmented image into patches
        patches = split_image_into_patches(image, self.num_patches)
        patch_tensors = [patch for patch in patches]  # Already in tensor form
        patch_tensors = torch.stack(patch_tensors)  # [num_patches, C, H, W]
        return patch_tensors, label


def get_dataloaders(
    train_df,
    test_df,
    root_dir,
    batch_size=16,
    config_path="config/config.yaml",
    num_patches=None,
):
    img_size = get_image_size_from_config(config_path)
    num_patches = get_num_patches_from_config(config_path, num_patches)
    train_transform = A.Compose(
        [
            A.OneOf(
                [
                    A.Downscale(
                        scale_range=(0.75, 0.75),
                        interpolation_pair={
                            "downscale": cv2.INTER_AREA,
                            "upscale": cv2.INTER_LINEAR,
                        },
                        p=0.1,
                    ),
                    A.Downscale(
                        scale_range=(0.75, 0.75),
                        interpolation_pair={
                            "downscale": cv2.INTER_AREA,
                            "upscale": cv2.INTER_LANCZOS4,
                        },
                        p=0.1,
                    ),
                    A.Downscale(
                        scale_range=(0.95, 0.95),
                        interpolation_pair={
                            "downscale": cv2.INTER_AREA,
                            "upscale": cv2.INTER_LINEAR,
                        },
                        p=0.8,
                    ),
                ],
                p=0.125,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.01, 0.08),
                hole_width_range=(0.01, 0.15),
                fill=0,
                p=0.1,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.1),
                rotate=(-30, 30),
                shear=(-10, 10),
                p=0.3,
            ),
            A.ElasticTransform(alpha=1, sigma=20, p=0.1),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2
            ),
            A.GaussNoise(p=0.1),
            A.Normalize([0.5] * 3, [0.5] * 3),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose([A.Normalize([0.5] * 3, [0.5] * 3), ToTensorV2()])
    train_dataset = CancerPatchDataset(train_df, root_dir, train_transform, num_patches)
    test_dataset = CancerPatchDataset(test_df, root_dir, test_transform, num_patches)
    class_counts = train_df["cancer"].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[train_df.iloc[i]["cancer"]] for i in range(total_samples)]
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=total_samples, replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
