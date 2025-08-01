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


def split_image_into_patches(image, num_patches=2, patch_size=None):
    """Split the original image into num_patches along the height with equal sizes."""
    # Handle both numpy.ndarray and torch.Tensor
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert [C, H, W] to [H, W, C]
    else:
        image = np.array(image)

    height, width = image.shape[:2]
    patch_height = height // num_patches
    if patch_size is None:
        patch_size = (patch_height, width)  # Default patch size

    patches = []
    for i in range(num_patches):
        start_h = i * patch_height
        end_h = (i + 1) * patch_height
        if i == num_patches - 1:
            end_h = height  # Include remaining pixels in the last patch
        patch = image[start_h:end_h, :, :]
        # Resize patch to ensure consistent size
        patch = cv2.resize(
            patch, (patch_size[1], patch_size[0]), interpolation=cv2.INTER_LINEAR
        )
        # Convert patch to tensor
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()  # [C, H_patch, W]
        patches.append(patch)
    return patches


class CancerPatchDataset(Dataset):
    def __init__(
        self, df, root_dir, transform=None, num_patches=2, augment_before_split=True
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.num_patches = num_patches
        self.img_size = get_image_size_from_config()
        self.augment_before_split = augment_before_split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])

        if self.augment_before_split:
            # Augment before splitting
            if self.transform:
                image_np = np.array(image)
                transform_with_resize = A.Compose(
                    [A.Resize(*self.img_size), *self.transform]
                )
                augmented = transform_with_resize(image=image_np)
                image = augmented["image"]
            else:
                image_np = np.array(image)
                image_np = cv2.resize(image_np, self.img_size)
                image = A.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image=image_np)[
                    "image"
                ]
                image = ToTensorV2()(image=image)["image"]

            patch_height = self.img_size[0] // self.num_patches
            patches = split_image_into_patches(
                image, self.num_patches, patch_size=(patch_height, self.img_size[1])
            )
            patch_tensors = torch.stack(patches)  # [num_patches, C, H_patch, W]
        else:
            # Split first, then augment
            patches = split_image_into_patches(
                image, self.num_patches
            )  # Returns list of [C, H_patch, W] tensors
            patch_tensors = []
            for patch in patches:
                # Convert tensor [C, H_patch, W] to NumPy [H_patch, W, C]
                patch_np = patch.permute(1, 2, 0).numpy()  # [H_patch, W, C]
                if self.transform:
                    transform_with_resize = A.Compose(
                        [A.Resize(*self.img_size), *self.transform]
                    )
                    augmented = transform_with_resize(image=patch_np)
                    patch = augmented["image"]  # [H, W, C] or tensor after ToTensorV2
                else:
                    patch_np = cv2.resize(patch_np, self.img_size)
                    patch = A.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image=patch_np)[
                        "image"
                    ]
                    patch = ToTensorV2()(image=patch)["image"]  # [C, H, W]
                patch_tensors.append(patch)
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
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # Increased for mammograms
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),  # Not suitable for grayscale mammograms
            # A.ToGray(p=0.2),  # Mammograms are already grayscale
            A.Equalize(p=0.3),  # Increased probability - very useful for mammograms
            # A.HEStain(p=0.2),  # Only for histopathology, not X-ray images
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),  # Modified for mammograms
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),  # Not suitable for grayscale
            # Additional augmentations good for mammograms:
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),  # Enhance edges/calcifications
            A.UnsharpMask(blur_limit=(3, 5), sigma_limit=(1.0, 2.0), alpha=(0.1, 0.3), p=0.2),  # Detail enhancement
            A.GaussNoise(p=0.1),
            A.Normalize([0.5] * 3, [0.5] * 3),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose([A.Normalize([0.5] * 3, [0.5] * 3), ToTensorV2()])
    train_dataset = CancerPatchDataset(
        train_df, root_dir, train_transform, num_patches, augment_before_split=True
    )
    test_dataset = CancerPatchDataset(
        test_df, root_dir, test_transform, num_patches, augment_before_split=True
    )
    class_counts = train_df["cancer"].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[train_df.iloc[i]["cancer"]] for i in range(total_samples)]
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=total_samples, replacement=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        #num_workers=16,  # Thử với 8 workers
        #pin_memory=True,  # Tăng tốc chuyển dữ liệu sang GPU
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,  # Thử với 8 workers
        pin_memory=True,  # Tăng tốc chuyển dữ liệu sang GPU
    )
    return train_loader, test_loader
