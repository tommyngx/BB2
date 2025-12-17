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
        img_size=None,  # Thêm img_size để biết kích thước cần resize
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
                # Ensure non-negative values
                x = max(0, float(row["x"]))
                y = max(0, float(row["y"]))
                w = max(0, float(row["width"]))
                h = max(0, float(row["height"]))

                # Get image dimensions for validation
                img_h, img_w = image.shape[:2]

                # Validate and clip bbox to image boundaries
                x = min(x, img_w - 1)
                y = min(y, img_h - 1)

                # Ensure bbox doesn't exceed image boundaries
                if x + w > img_w:
                    w = img_w - x
                if y + h > img_h:
                    h = img_h - y

                # Final check: bbox must have positive area
                if w > 0 and h > 0:
                    # Convert to x1, y1, x2, y2 format (pascal_voc)
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    bbox = [x1, y1, x2, y2]
                    has_bbox = True
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    has_bbox = False
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
                has_bbox = False
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
            has_bbox = False

        # Apply transforms based on has_bbox
        if has_bbox:
            # Positive sample with valid bbox - use bbox-safe transform
            if self.positive_transform:
                # Pass bbox to transform - Albumentations will update it
                transformed = self.positive_transform(
                    image=image, bboxes=[bbox], labels=[label]
                )
                image = transformed["image"]

                # Get the TRANSFORMED bbox from Albumentations
                if len(transformed["bboxes"]) > 0:
                    # Albumentations returns transformed bbox coordinates
                    transformed_bbox = list(transformed["bboxes"][0])

                    # Validate transformed bbox
                    # After resize to img_size, bbox should be in pixel coordinates
                    # relative to the resized image
                    if self.img_size is not None:
                        h_img, w_img = self.img_size

                        # Check format: x2 > x1 and y2 > y1
                        if (
                            transformed_bbox[2] > transformed_bbox[0]
                            and transformed_bbox[3] > transformed_bbox[1]
                        ):
                            # Clip to image bounds (just to be safe)
                            transformed_bbox[0] = max(
                                0.0, min(float(w_img), transformed_bbox[0])
                            )
                            transformed_bbox[1] = max(
                                0.0, min(float(h_img), transformed_bbox[1])
                            )
                            transformed_bbox[2] = max(
                                0.0, min(float(w_img), transformed_bbox[2])
                            )
                            transformed_bbox[3] = max(
                                0.0, min(float(h_img), transformed_bbox[3])
                            )

                            # Recheck after clipping
                            if (
                                transformed_bbox[2] > transformed_bbox[0]
                                and transformed_bbox[3] > transformed_bbox[1]
                            ):
                                bbox = transformed_bbox
                            else:
                                bbox = [0.0, 0.0, 0.0, 0.0]
                                has_bbox = False
                        else:
                            bbox = [0.0, 0.0, 0.0, 0.0]
                            has_bbox = False
                    else:
                        # No img_size specified, just validate format
                        if (
                            transformed_bbox[2] > transformed_bbox[0]
                            and transformed_bbox[3] > transformed_bbox[1]
                        ):
                            bbox = transformed_bbox
                        else:
                            bbox = [0.0, 0.0, 0.0, 0.0]
                            has_bbox = False
                else:
                    # Transform removed bbox (min_visibility or out of bounds)
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    has_bbox = False
        else:
            # Negative sample - do NOT pass bbox to transform
            if self.negative_transform is not None:
                transformed = self.negative_transform(image=image)
                image = transformed["image"]
            else:
                # Test mode fallback
                from .m2_augment import get_m2_test_augmentation_no_bbox

                if self.img_size is not None:
                    if isinstance(self.img_size, (list, tuple)):
                        h_target, w_target = self.img_size
                    else:
                        h_target = w_target = self.img_size
                else:
                    h_target = w_target = 448

                simple_transform = get_m2_test_augmentation_no_bbox(h_target, w_target)
                transformed = simple_transform(image=image)
                image = transformed["image"]

        # Normalize bbox to [0, 1] range based on TRANSFORMED image size
        if has_bbox and bbox != [0.0, 0.0, 0.0, 0.0]:
            # Image after ToTensorV2 is [C, H, W]
            # bbox is in pixel coordinates relative to transformed image
            if self.img_size is not None:
                h_img, w_img = self.img_size
            else:
                # Fallback: get from tensor shape
                h_img, w_img = image.shape[1], image.shape[2]

            # Normalize to [0, 1]
            bbox_normalized = [
                bbox[0] / w_img,  # x1
                bbox[1] / h_img,  # y1
                bbox[2] / w_img,  # x2
                bbox[3] / h_img,  # y2
            ]

            # Clip to [0, 1] range
            bbox_normalized = [max(0.0, min(1.0, coord)) for coord in bbox_normalized]

            # Final validation
            if (
                bbox_normalized[2] <= bbox_normalized[0]
                or bbox_normalized[3] <= bbox_normalized[1]
            ):
                bbox = [0.0, 0.0, 0.0, 0.0]
                has_bbox = False
            else:
                # Check minimum size (1% of image)
                min_bbox_size = 0.01
                bbox_width = bbox_normalized[2] - bbox_normalized[0]
                bbox_height = bbox_normalized[3] - bbox_normalized[1]

                if bbox_width < min_bbox_size or bbox_height < min_bbox_size:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    has_bbox = False
                else:
                    bbox = bbox_normalized

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
        img_size=(height, width),  # Truyền img_size vào
    )

    # Test dataset: dùng cùng một transform cho cả positive và negative
    test_dataset = M2Dataset(
        test_df,
        data_folder,
        positive_transform=test_transform,
        negative_transform=None,
        mode="test",
        img_size=(height, width),  # Truyền img_size vào
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
