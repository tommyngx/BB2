import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from .dataloader import get_weighted_sampler, load_metadata
from .m2_augment import get_m2_train_augmentation, get_m2_test_augmentation


class M2Dataset(Dataset):
    """Dataset for multi-task learning: classification + bbox regression"""

    def __init__(
        self,
        df,
        data_folder,
        positive_transform=None,
        negative_transform=None,
        mode="train",
    ):
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.positive_transform = positive_transform  # For images with bbox
        self.negative_transform = negative_transform  # For images without bbox
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_folder, row["link"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row["cancer"])

        # Get bounding box if positive, else set to zeros
        if label == 1 and all(
            col in row.index for col in ["x", "y", "width", "height"]
        ):
            # Check if bbox values are valid (not NaN)
            if (
                pd.notna(row["x"])
                and pd.notna(row["y"])
                and pd.notna(row["width"])
                and pd.notna(row["height"])
            ):
                x, y, w, h = row["x"], row["y"], row["width"], row["height"]
                # Convert to x1, y1, x2, y2 format
                x1, y1 = float(x), float(y)
                x2, y2 = float(x + w), float(y + h)
                bbox = [x1, y1, x2, y2]
                has_bbox = True
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
                has_bbox = False
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
            has_bbox = False

        # Apply different transforms based on has_bbox
        if has_bbox and self.positive_transform:
            # Positive sample - use bbox-safe transform
            transformed = self.positive_transform(
                image=image, bboxes=[bbox], labels=[label]
            )
            image = transformed["image"]
            # Get transformed bbox
            if len(transformed["bboxes"]) > 0:
                bbox = list(transformed["bboxes"][0])
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
                has_bbox = False  # bbox was lost during augmentation
        elif not has_bbox and self.negative_transform:
            # Negative sample - use aggressive transform (no bbox)
            transformed = self.negative_transform(image=image)
            image = transformed["image"]
        elif self.positive_transform:
            # Fallback to positive transform with empty bbox
            transformed = self.positive_transform(
                image=image, bboxes=[bbox], labels=[label]
            )
            image = transformed["image"]

        # Normalize bbox to [0, 1] range based on image size
        if has_bbox and bbox != [0.0, 0.0, 0.0, 0.0]:
            h_img, w_img = image.shape[1], image.shape[2]  # CHW format after ToTensorV2
            bbox = [bbox[0] / w_img, bbox[1] / h_img, bbox[2] / w_img, bbox[3] / h_img]

        return {
            "image": image,
            "label": label,
            "bbox": np.array(bbox, dtype=np.float32),
            "has_bbox": int(has_bbox),
        }


def get_m2_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    img_size=None,
    mode="train",
):
    """
    Get dataloaders for multi-task learning
    """
    from .dataloader import get_image_size_from_config

    if img_size is None:
        img_size = get_image_size_from_config(config_path)

    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = img_size

    # Get transforms
    positive_train_transform, negative_train_transform = get_m2_train_augmentation(
        height, width
    )
    test_transform = get_m2_test_augmentation(height, width)

    # Create datasets
    train_dataset = M2Dataset(
        train_df,
        data_folder,
        positive_transform=positive_train_transform,
        negative_transform=negative_train_transform,
        mode="train",
    )
    test_dataset = M2Dataset(
        test_df,
        data_folder,
        positive_transform=test_transform,
        negative_transform=test_transform,  # Test không cần phân biệt
        mode="test",
    )

    # Create dataloaders
    num_workers = 0 if mode == "test" else 4

    # Use weighted sampler for training
    sampler = get_weighted_sampler(train_df, label_col="cancer")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
