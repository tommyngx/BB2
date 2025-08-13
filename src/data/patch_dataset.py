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
from .based_data import get_num_workers
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import multiprocessing


def split_image_into_patches(image, num_patches=2, patch_size=None, overlap_ratio=0.2):
    """
    Split an image into vertical patches with optional overlap.

    Args:
        image (PIL.Image, np.ndarray, or torch.Tensor): Input image.
        num_patches (int): Number of patches to split.
        patch_size (tuple or None): Target patch size (height, width). If None, calculated automatically.
        overlap_ratio (float): Overlap ratio between consecutive patches.

    Returns:
        list of torch.Tensor: List of patch tensors (C, H, W).
    """
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


def compute_required_img_shape(img_size, num_patches, overlap_ratio=0.2):
    """
    Compute the required original image shape so that after splitting into patches (with overlap),
    each patch can be resized to img_size.

    Args:
        img_size (tuple): (height, width) of the final patch.
        num_patches (int): Number of patches.
        overlap_ratio (float): Overlap ratio between patches.

    Returns:
        tuple: (required_height, required_width)
    """
    patch_height, patch_width = img_size
    step = int(patch_height * (1 - overlap_ratio))
    required_height = step * (num_patches - 1) + patch_height
    required_width = patch_width
    return (required_height, required_width)


class CancerPatchDataset(Dataset):
    def __init__(
        self,
        df,
        data_folder,
        transform=None,
        num_patches=2,
        augment_before_split=True,
        config_path="config/config.yaml",
        img_size=None,
        overlap_ratio=0.2,
    ):
        """
        Dataset for splitting images into patches for MIL training.

        Args:
            df (pd.DataFrame): DataFrame with image paths and labels.
            data_folder (str): Root directory containing images.
            transform (albumentations.Compose or None): Augmentation pipeline.
            num_patches (int): Number of patches per image.
            augment_before_split (bool): Whether to augment before splitting.
            config_path (str): Path to config file.
            img_size (tuple or None): Target patch size (height, width).
            overlap_ratio (float): Overlap ratio between patches.
        """
        if data_folder is None:
            raise ValueError(
                "data_folder must not be None. Please provide the data folder path."
            )
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transform = transform
        self.num_patches = num_patches
        if img_size is None:
            self.img_size = get_image_size_from_config(config_path)
        else:
            self.img_size = img_size
        self.augment_before_split = augment_before_split
        self.overlap_ratio = overlap_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            patch_tensors (torch.Tensor): Tensor of shape (num_patches+1, C, H, W).
            label (int): Image label.
        """
        img_path = os.path.join(self.data_folder, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])

        # Ensure img_size is (height, width) and both are int > 0
        img_size_hw = self.img_size
        if isinstance(img_size_hw, (list, tuple)):
            if len(img_size_hw) >= 2:
                h, w = int(img_size_hw[0]), int(img_size_hw[1])
            else:
                h = w = 448  # fallback
        else:
            h = w = int(img_size_hw) if img_size_hw else 448

        # Ensure positive values
        if h <= 0 or w <= 0:
            h = w = 448

        # Store original image for later use as full patch
        original_image = np.array(image)

        # --- Resize image to required shape before augmentation ---
        required_shape = compute_required_img_shape(
            (h, w), self.num_patches, self.overlap_ratio
        )
        image = image.resize(
            (required_shape[1], required_shape[0]), resample=Image.BILINEAR
        )

        # Apply augmentation to the resized image (no resize in this step)
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented["image"]
        else:
            image = np.array(image)

        # Split the augmented image into patches
        patches = split_image_into_patches(
            image, self.num_patches, overlap_ratio=self.overlap_ratio
        )
        patch_tensors = []

        # Create normalization and tensor conversion pipeline
        normalize_and_tensorize = A.Compose(
            [A.Normalize(mean=[0.5] * 3, std=[0.5] * 3), ToTensorV2()]
        )

        # Resize each patch to the standard size and convert to tensor
        for patch in patches:
            patch_np = patch.permute(1, 2, 0).numpy().astype(np.uint8)
            patch_np = cv2.resize(patch_np, (w, h))
            patch_tensor = normalize_and_tensorize(image=patch_np)["image"]
            patch_tensors.append(patch_tensor)

        # Add full AUGMENTED image as the last patch (resize to img_size, normalize, to tensor)
        # Use augmented image (after transform), not the original image
        augmented_image = image if isinstance(image, np.ndarray) else np.array(image)
        print(
            f"[DEBUG] full_img_resized: w={w}, h={h}, augmented_image.shape={augmented_image.shape}"
        )
        full_img_resized = cv2.resize(augmented_image, (w, h))
        full_img_tensor = normalize_and_tensorize(image=full_img_resized)["image"]
        patch_tensors.append(full_img_tensor)

        patch_tensors = torch.stack(patch_tensors)
        return patch_tensors, label


def get_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    num_patches=None,
    num_workers=None,
    pin_memory=True,
    img_size=None,
    overlap_ratio=0.2,
):
    if num_workers is None:
        num_workers = get_num_workers()
    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    num_patches = get_num_patches_from_config(config_path, num_patches)
    # Ensure img_size is a tuple (height, width) with int values
    if isinstance(img_size, (list, tuple)):
        height, width = img_size
        if height is None or width is None:
            height = width = 448
        height, width = int(height), int(width)
    else:
        height = width = int(img_size)
    if height is None or width is None:
        height = width = 448
    # Only pass height, width to get_train_augmentation if both are not None
    if height is not None and width is not None:
        train_transform = get_train_augmentation(height, width, resize_first=False)
        test_transform = get_test_augmentation(height, width, resize_first=False)
    else:
        # fallback: no resize if height/width are not valid
        train_transform = get_train_augmentation(448, 448, resize_first=False)
        test_transform = get_test_augmentation(448, 448, resize_first=False)
    # Use the same img_size for both train and test datasets
    train_dataset = CancerPatchDataset(
        train_df,
        data_folder,
        train_transform,
        num_patches,
        augment_before_split=True,
        config_path=config_path,
        img_size=(height, width),
        overlap_ratio=overlap_ratio,
    )
    test_dataset = CancerPatchDataset(
        test_df,
        data_folder,
        test_transform,
        num_patches,
        augment_before_split=True,
        config_path=config_path,
        img_size=(height, width),
        overlap_ratio=overlap_ratio,
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


"""
Explanation of the patch augmentation and resizing pipeline:

- When loading an image, it is opened with PIL and converted to RGB.
- The image is resized to the required shape (computed based on the number of patches, overlap, and output img_size) before augmentation.
- If a transform (augmentation) is provided, it is applied to the resized image.
    - Note: The augmentation pipeline always includes a final resize step (because get_train_augmentation and get_test_augmentation always append A.Resize(height, width) at the end if resize_first=False).
    - Therefore, the image will be resized again to (height, width) after augmentation.
- After augmentation, the image is split into patches using `split_image_into_patches`.
- Each patch is then resized to the standard size (e.g., 448x448 or as specified) using `cv2.resize`.
- After resizing, each patch is normalized (`A.Normalize`) and converted to a tensor (`ToTensorV2`).
- The output is a stacked tensor of all patches.

Summary:
- The original image is resized to be large enough for patch splitting (according to the number of patches and overlap).
- Augmentation (transform) always includes a resize to (height, width) at the end of the pipeline (by design of get_train_augmentation).
- Each patch is resized again after splitting, ensuring all output patches have the same size (e.g., 448x448).
- The final patch size is always the standard size specified by argument or config.

Note:
- If you want to augment without resizing in the pipeline, modify get_train_augmentation to avoid adding A.Resize(height, width) when resize_first=False.

    - Each patch is resized to 448x448.
    - Each patch is normalized and converted to a tensor.

Advantages:
- Ensures all output patches have the same size, suitable for patch-based models.
- Augmentation does not alter the original spatial information before splitting.
- Saves RAM by resizing the original image before augmentation and splitting.

If you split an image into 3 vertical patches (`p3`), each patch will have a size of **448x448** (after resizing each patch), so:

- **The minimum required original image height** depends on the patch splitting and overlap.
- If no overlap (`overlap_ratio=0`):
  - Minimum original image height = 3 × 448 = **1344** pixels (width is 448).
- If overlap is 20% (`overlap_ratio=0.2`):
  - Each patch height is 448, step between patches is `step = 448 × (1 - 0.2) = 358.4 ≈ 358` pixels.
  - Minimum original image height = `step × (num_patches - 1) + patch_height = 358 × 2 + 448 = 1164` pixels (width is 448).

Summary:
- No overlap: minimum image size is 1344x448.
- 20% overlap: minimum image size is about 1164x448.
- After splitting, each patch is always resized to 448x448.

Note:
- If the original image is smaller, the last patch may repeat regions or be resized to ensure the correct number of patches.
"""
