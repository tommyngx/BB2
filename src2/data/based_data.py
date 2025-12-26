import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from .augment import (
    get_train_augmentation_positive,
    get_train_augmentation_negative,
    get_test_augmentation_positive,
    get_test_augmentation_negative,
)
from .dataloader import get_image_size_from_config, get_weighted_sampler
import multiprocessing


def get_num_workers():
    # Trả về số worker tối thiểu là 2, tối đa là 4
    try:
        cpu_count = multiprocessing.cpu_count()
        return max(2, min(6, cpu_count))
    except Exception:
        return 2


def validate_and_clip_bbox(x, y, w, h, img_w, img_h, min_area=100):
    """Validate and clip bbox to image boundaries"""
    x = max(0, float(x))
    y = max(0, float(y))
    w = max(0, float(w))
    h = max(0, float(h))
    x = min(x, img_w - 1)
    y = min(y, img_h - 1)
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    is_valid = w > 0 and h > 0 and (w * h) >= min_area
    return x, y, w, h, is_valid


class BasicImageDataset(Dataset):
    def __init__(
        self,
        df,
        data_folder,
        positive_transform=None,
        negative_transform=None,
        mode="train",
        img_size=None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.positive_transform = positive_transform
        self.negative_transform = negative_transform
        self.mode = mode
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_folder, row["link"])
        image = cv2.imread(img_path)
        if image is None:
            h, w = self.img_size if self.img_size else (448, 448)
            image = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w = image.shape[:2]
        label = int(row["cancer"])

        # Get bbox_list from preprocessed df (already grouped)
        bbox_list = row.get("bbox_list", [])
        if not isinstance(bbox_list, list):
            bbox_list = []

        # Validate and clip bboxes
        valid_bboxes = []
        for bbox in bbox_list:
            x, y, w, h = bbox
            x, y, w, h, is_valid = validate_and_clip_bbox(
                x, y, w, h, img_w, img_h, min_area=100
            )
            if is_valid:
                valid_bboxes.append([x, y, w, h])

        # Apply augmentation: ALWAYS use bbox if available (both train and test)
        if len(valid_bboxes) > 0 and self.positive_transform:
            # Positive: use bbox-safe augmentation (train và test đều cần bbox)
            bbox_labels = [1] * len(valid_bboxes)
            transformed = self.positive_transform(
                image=image, bboxes=valid_bboxes, labels=bbox_labels
            )
            image = transformed["image"]
        elif len(valid_bboxes) == 0 and self.negative_transform:
            # Negative: use aggressive augmentation (no bbox)
            transformed = self.negative_transform(image=image)
            image = transformed["image"]
        else:
            raise RuntimeError("No valid transform found for this sample.")

        return {
            "image": image,
            "label": label,
            "image_id": str(row["image_id"]),
        }


def get_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    img_size=None,
    num_workers=None,
    pin_memory=True,
    mode="train",
):
    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    if isinstance(img_size, (list, tuple)):
        height, width = int(img_size[0]), int(img_size[1])
    else:
        height = width = int(img_size)
    if num_workers is None:
        num_workers = 0 if mode == "test" else get_num_workers()

    # Augmentations
    positive_train_transform = get_train_augmentation_positive(height, width)
    negative_train_transform = get_train_augmentation_negative(height, width)
    test_transform = get_test_augmentation_positive(height, width)
    test_transform_neg = get_test_augmentation_negative(height, width)

    train_dataset = BasicImageDataset(
        train_df,
        data_folder,
        positive_transform=positive_train_transform,
        negative_transform=negative_train_transform,
        mode="train",
        img_size=(height, width),
    )
    test_dataset = BasicImageDataset(
        test_df,
        data_folder,
        positive_transform=test_transform,
        negative_transform=test_transform_neg,
        mode="test",
        img_size=(height, width),
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
