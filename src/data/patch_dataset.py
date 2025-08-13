import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .augment import get_train_augmentation, get_test_augmentation
from .dataloader import (
    get_image_size_from_config,
    get_num_patches_from_config,
    get_weighted_sampler,
)
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A


def split_image_into_patches(image, num_patches=2, patch_size=None, overlap_ratio=0.2):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    else:
        image = np.array(image)
    height, width = image.shape[:2]
    if patch_size is None:
        patch_height = height // num_patches
        patch_size = (patch_height, width)
    else:
        patch_height = patch_size[0]
    step = int(patch_height * (1 - overlap_ratio))
    patches = []
    for i in range(num_patches):
        start_h = i * step
        end_h = start_h + patch_height
        if end_h > height:
            end_h = height
            start_h = max(0, end_h - patch_height)
        patch = image[start_h:end_h, :, :]
        patch = cv2.resize(
            patch, (patch_size[1], patch_size[0]), interpolation=cv2.INTER_LINEAR
        )
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        patches.append(patch)
    return patches


class CancerPatchDataset(Dataset):
    def __init__(
        self,
        df,
        data_folder,  # đã đổi tên từ root_dir sang data_folder
        transform=None,
        num_patches=2,
        augment_before_split=True,
        config_path="config/config.yaml",
    ):
        if data_folder is None:
            raise ValueError(
                "data_folder must not be None. Please provide the data folder path."
            )
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transform = transform
        self.num_patches = num_patches
        self.img_size = get_image_size_from_config(config_path)
        self.augment_before_split = augment_before_split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])
        if self.augment_before_split:
            if self.transform:
                image_np = np.array(image)
                augmented = self.transform(image=image_np)
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
            patch_tensors = torch.stack(patches)
        else:
            patches = split_image_into_patches(image, self.num_patches)
            patch_tensors = []
            for patch in patches:
                patch_np = patch.permute(1, 2, 0).numpy()
                if self.transform:
                    augmented = self.transform(image=patch_np)
                    patch = augmented["image"]
                else:
                    patch_np = cv2.resize(patch_np, self.img_size)
                    patch = A.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image=patch_np)[
                        "image"
                    ]
                    patch = ToTensorV2()(image=patch)["image"]
                patch_tensors.append(patch)
            patch_tensors = torch.stack(patch_tensors)
        return patch_tensors, label


def get_dataloaders(
    train_df,
    test_df,
    data_folder,  # đã đổi tên từ root_dir sang data_folder
    batch_size=16,
    config_path="config/config.yaml",
    num_patches=None,
    num_workers=4,
    pin_memory=True,
    img_size=None,
):
    # Nếu img_size được truyền vào thì dùng, nếu không thì lấy từ config
    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    num_patches = get_num_patches_from_config(config_path, num_patches)
    if isinstance(img_size, (list, tuple)):
        height, width = int(img_size[0]), int(img_size[1])
    else:
        height = width = int(img_size)
    train_transform = get_train_augmentation(height, width, resize_first=False)
    test_transform = get_test_augmentation(height, width)
    train_dataset = CancerPatchDataset(
        train_df,
        data_folder,
        train_transform,
        num_patches,
        augment_before_split=True,
        config_path=config_path,
    )
    test_dataset = CancerPatchDataset(
        test_df,
        data_folder,
        test_transform,
        num_patches,
        augment_before_split=True,
        config_path=config_path,
    )
    sampler = get_weighted_sampler(train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
