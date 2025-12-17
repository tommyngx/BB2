import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from .dataloader import get_weighted_sampler, load_metadata
from .m2_augment import get_m2_train_augmentation, get_m2_test_augmentation


def validate_and_clip_bbox(x, y, w, h, img_w, img_h, min_area=100):
    """
    Validate and clip bbox to image boundaries
    Returns: (x, y, w, h, is_valid)
    """
    # Ensure non-negative
    x = max(0, float(x))
    y = max(0, float(y))
    w = max(0, float(w))
    h = max(0, float(h))

    # Clip to image boundaries
    x = min(x, img_w - 1)
    y = min(y, img_h - 1)

    # Ensure bbox doesn't exceed boundaries
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y

    # Check validity: positive area and min size
    is_valid = w > 0 and h > 0 and (w * h) >= min_area

    return x, y, w, h, is_valid


class M2DETRDataset(Dataset):
    """
    Dataset for DETR-style multi-bbox detection
    Returns: image, label, list of bboxes (variable length)
    """

    def __init__(
        self,
        df,
        data_folder,
        positive_transform=None,
        negative_transform=None,
        mode="train",
        img_size=None,
        max_objects=5,  # Reduced from 10 to 5 for mammography
    ):
        self.data_folder = data_folder
        self.mode = mode
        self.img_size = img_size
        self.max_objects = max_objects
        self.positive_transform = positive_transform
        self.negative_transform = negative_transform

        # Group by image_id to get all bboxes per image
        self.samples = self._prepare_samples(df)

    def _prepare_samples(self, df):
        """Group bboxes by image_id with proper validation"""
        samples = []
        multi_bbox_count = 0  # Count images with >1 bbox

        for image_id in df["image_id"].unique():
            img_rows = df[df["image_id"] == image_id]
            first_row = img_rows.iloc[0]

            img_path = os.path.join(self.data_folder, first_row["link"])
            label = int(first_row["cancer"])

            bboxes = []
            if label == 1:
                try:
                    temp_img = cv2.imread(img_path)
                    if temp_img is not None:
                        img_h, img_w = temp_img.shape[:2]
                    else:
                        print(f"⚠️ Cannot read image: {img_path}")
                        continue
                except Exception as e:
                    print(f"⚠️ Error reading {img_path}: {e}")
                    continue

                # Process ALL rows for this image_id (multi-bbox support)
                for _, row in img_rows.iterrows():
                    if all(col in row.index for col in ["x", "y", "width", "height"]):
                        if all(
                            pd.notna(row[col]) for col in ["x", "y", "width", "height"]
                        ):
                            x, y, w, h, is_valid = validate_and_clip_bbox(
                                row["x"],
                                row["y"],
                                row["width"],
                                row["height"],
                                img_w,
                                img_h,
                                min_area=100,
                            )
                            if is_valid:
                                bboxes.append([x, y, w, h])

                # Debug: count images with multiple bboxes
                if len(bboxes) > 1:
                    multi_bbox_count += 1
                    if multi_bbox_count <= 5:  # Print first 5 multi-bbox images
                        print(f"[DEBUG] image_id={image_id} has {len(bboxes)} bboxes")

            samples.append(
                {
                    "image_path": img_path,
                    "label": label,
                    "bboxes": bboxes,
                    "image_id": image_id,
                }
            )

        print(f"[INFO] Total images with >1 bbox: {multi_bbox_count}/{len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"])
        if image is None:
            # Fallback: return dummy data
            print(f"⚠️ Failed to load image: {sample['image_path']}")
            dummy_img = np.zeros((448, 448, 3), dtype=np.uint8)
            return {
                "image": torch.zeros(3, 448, 448),
                "label": 0,
                "bboxes": torch.zeros(self.max_objects, 4),
                "bbox_mask": torch.zeros(self.max_objects),
                "num_objects": 0,
                "image_id": sample["image_id"],
            }

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        label = sample["label"]
        bboxes = sample["bboxes"].copy()  # List of [x, y, w, h] already validated

        # Apply augmentation
        if len(bboxes) > 0:
            # Positive sample: use bbox-aware transform
            if self.positive_transform:
                # Albumentations expects list of bboxes and labels
                bbox_labels = [1] * len(bboxes)

                transformed = self.positive_transform(
                    image=image, bboxes=bboxes, labels=bbox_labels
                )
                image = transformed["image"]
                bboxes = list(
                    transformed["bboxes"]
                )  # Updated bboxes after augmentation
        else:
            # Negative sample: use aggressive transform without bbox
            if self.negative_transform:
                transformed = self.negative_transform(image=image)
                image = transformed["image"]
            else:
                # Fallback for test mode
                from .m2_augment import get_m2_test_augmentation_no_bbox

                if self.img_size:
                    h_t, w_t = (
                        self.img_size
                        if isinstance(self.img_size, tuple)
                        else (self.img_size, self.img_size)
                    )
                else:
                    h_t, w_t = 448, 448
                simple_transform = get_m2_test_augmentation_no_bbox(h_t, w_t)
                transformed = simple_transform(image=image)
                image = transformed["image"]

        # Get target image size after transform
        if self.img_size:
            h_img, w_img = self.img_size
        else:
            h_img, w_img = image.shape[1], image.shape[2]

        # Normalize and validate bboxes to [0, 1], then pad to max_objects
        normalized_bboxes = []
        for bbox in bboxes[: self.max_objects]:  # Limit to max_objects
            x, y, w, h = bbox

            # Re-validate after augmentation (bbox might be out of bounds)
            x, y, w, h, is_valid = validate_and_clip_bbox(
                x,
                y,
                w,
                h,
                w_img,
                h_img,
                min_area=1,  # Allow small bboxes after resize
            )

            if not is_valid:
                continue

            # Normalize to [0, 1]
            norm_bbox = [
                max(0.0, min(1.0, x / w_img)),
                max(0.0, min(1.0, y / h_img)),
                max(0.0, min(1.0, w / w_img)),
                max(0.0, min(1.0, h / h_img)),
            ]

            # Final check: min 1% size
            if norm_bbox[2] > 0.01 and norm_bbox[3] > 0.01:
                normalized_bboxes.append(norm_bbox)

        # Pad to max_objects
        num_objects = len(normalized_bboxes)
        padded_bboxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        if num_objects > 0:
            padded_bboxes[:num_objects] = np.array(normalized_bboxes, dtype=np.float32)

        # Create mask: 1 for valid bbox, 0 for padding
        bbox_mask = np.zeros(self.max_objects, dtype=np.float32)
        bbox_mask[:num_objects] = 1.0

        return {
            "image": image,
            "label": label,
            "bboxes": torch.from_numpy(padded_bboxes),  # [max_objects, 4]
            "bbox_mask": torch.from_numpy(bbox_mask),  # [max_objects]
            "num_objects": num_objects,
            "image_id": sample["image_id"],
        }


def get_m2_detr_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    img_size=None,
    mode="train",
    max_objects=5,  # Default 5 for mammography
):
    """Get DETR-style dataloaders"""
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
    train_dataset = M2DETRDataset(
        train_df,
        data_folder,
        positive_transform=positive_train_transform,
        negative_transform=negative_train_transform,
        mode="train",
        img_size=(height, width),
        max_objects=max_objects,
    )

    test_dataset = M2DETRDataset(
        test_df,
        data_folder,
        positive_transform=test_transform,
        negative_transform=None,
        mode="test",
        img_size=(height, width),
        max_objects=max_objects,
    )

    # Dataloaders
    num_workers = 0 if mode == "test" else 4
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
