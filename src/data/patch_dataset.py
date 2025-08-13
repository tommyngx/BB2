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
    Tính kích thước ảnh gốc cần thiết để sau khi chia patch (có overlap) mỗi patch resize về img_size.
    img_size: tuple (h, w) là kích thước patch sau cùng.
    num_patches: số patch muốn chia.
    overlap_ratio: tỉ lệ overlap giữa các patch.
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
        img_path = os.path.join(self.data_folder, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])
        # --- Resize ảnh về kích thước cần thiết trước khi augment ---
        required_shape = compute_required_img_shape(
            self.img_size, self.num_patches, self.overlap_ratio
        )
        image = image.resize(
            (required_shape[1], required_shape[0]), resample=Image.BILINEAR
        )
        # Augment ảnh gốc (không resize ở bước này)
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented["image"]
        else:
            image = np.array(image)
        # Split ảnh đã augment thành các patch
        patches = split_image_into_patches(
            image, self.num_patches, overlap_ratio=self.overlap_ratio
        )
        patch_tensors = []
        # Resize từng patch về kích thước tiêu chuẩn
        for patch in patches:
            patch_np = patch.permute(1, 2, 0).numpy()
            patch_np = cv2.resize(patch_np, self.img_size)
            patch = A.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image=patch_np)["image"]
            patch = ToTensorV2()(image=patch)["image"]
            patch_tensors.append(patch)
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
    # Đảm bảo img_size là tuple (height, width) với giá trị int
    if isinstance(img_size, (list, tuple)):
        height, width = img_size
        if height is None or width is None:
            height = width = 448
        height, width = int(height), int(width)
    else:
        height = width = int(img_size)
    if height is None or width is None:
        height = width = 448
    # Chỉ truyền height, width vào get_train_augmentation nếu cả hai đều khác None
    if height is not None and width is not None:
        train_transform = get_train_augmentation(height, width, resize_first=False)
        test_transform = get_test_augmentation(height, width)
    else:
        # fallback: không resize nếu height/width không hợp lệ
        train_transform = get_train_augmentation(448, 448, resize_first=False)
        test_transform = get_test_augmentation(448, 448)
    # Truyền cùng img_size cho cả train và test dataset
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
Giải thích cách xử lý augment và resize patch với code hiện tại:

- Khi lấy một ảnh từ đường dẫn, ảnh sẽ được mở bằng PIL và chuyển sang RGB.
- Nếu có transform (augmentation), ảnh gốc sẽ được augment trước (không resize ở bước này vì truyền height, width là None).
- Sau khi augment, ảnh sẽ được chia thành các patch bằng hàm `split_image_into_patches`.
- Mỗi patch sau khi chia sẽ được resize về kích thước tiêu chuẩn (ví dụ: 448x448 hoặc kích thước truyền vào qua argument/config) bằng `cv2.resize`.
- Sau khi resize, patch sẽ được chuẩn hóa (`A.Normalize`) và chuyển sang tensor (`ToTensorV2`).
- Kết quả trả về là một tensor stack gồm các patch đã resize, chuẩn hóa, chuyển sang tensor.

Tóm lại:
- Augment (transform) chỉ thực hiện các phép biến đổi như flip, rotate, v.v. nhưng không resize.
- Việc resize về kích thước chuẩn được thực hiện sau khi chia patch, áp dụng cho từng patch riêng biệt.
- Kích thước patch cuối cùng luôn là kích thước chuẩn (ví dụ: 448x448) lấy từ argument hoặc config.

Ví dụ:
- Nếu ảnh gốc là 1024x1024, img_size=(448,448), num_patches=3:
    - Ảnh gốc được augment (không resize).
    - Ảnh augment xong được chia thành 3 patch dọc theo chiều cao (có overlap).
    - Mỗi patch được resize về 448x448.
    - Patch được chuẩn hóa và chuyển sang tensor.

Ưu điểm:  
- Đảm bảo mọi patch đầu ra đều có cùng kích thước, phù hợp với input của model patch.
- Augment không làm thay đổi kích thước ảnh gốc, giúp giữ nguyên thông tin spatial trước khi chia patch.

Nếu bạn chia ảnh thành 3 patch dọc theo chiều cao (`p3`), mỗi patch có kích thước **448x448** (sau khi resize từng patch), thì:

- **Kích thước ảnh gốc trước khi chia patch** sẽ phụ thuộc vào cách chia patch và overlap.
- Nếu không overlap (`overlap_ratio=0`):  
  - Tổng chiều cao ảnh gốc tối thiểu = 3 × 448 = **1344** pixels (chiều rộng là 448).
- Nếu có overlap (ví dụ mặc định `overlap_ratio=0.2`):  
  - Mỗi patch cao 448, bước nhảy giữa các patch là `step = 448 × (1 - 0.2) = 358.4 ≈ 358` pixels.
  - Tổng chiều cao ảnh gốc tối thiểu = `step × (num_patches - 1) + patch_height = 358 × 2 + 448 = 1164` pixels (chiều rộng là 448).

**Tóm lại:**
- Nếu không overlap: ảnh gốc tối thiểu 1344x448.
- Nếu overlap 20%: ảnh gốc tối thiểu khoảng 1164x448.
- Sau khi chia, mỗi patch luôn được resize về 448x448.

**Lưu ý:**  
- Nếu ảnh gốc nhỏ hơn, patch cuối cùng sẽ bị lặp lại vùng ảnh hoặc resize lại để đủ số patch.
"""
