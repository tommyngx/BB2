import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


def get_image_size_from_config(config_path="config/config.yaml"):
    import yaml
    import os

    # Nếu config_path là tên file, tìm trong thư mục config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    # Hỗ trợ cả kiểu int hoặc list/tuple
    img_size = config.get("image_size", 448)
    if isinstance(img_size, (list, tuple)):
        return tuple(img_size)
    return (img_size, img_size)


def get_target_column_from_config(config_path="config/config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("target_column", "cancer")


def load_data(data_folder, config_path="config/config.yaml"):
    target_column = get_target_column_from_config(config_path)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    # df = df.dropna()
    df = df.drop_duplicates(subset=["link"])
    # Nếu cột target là string (label), chuyển thành số nguyên liên tục
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
    # Nếu có nhiều hơn 2 class, in ra danh sách class
    if train_df[label_col].nunique() > 2:
        print(
            "Detected multi-class classification. Classes:",
            list(class_names),
        )
    # Đảm bảo downstream dùng đúng cột label số
    train_df["cancer"] = train_df[label_col]
    test_df["cancer"] = test_df[label_col]
    return train_df, test_df


def load_images_and_labels(df, data_folder):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(data_folder, row["link"])
        img = pd.read_csv(img_path) if img_path.endswith(".csv") else None
        images.append(img)
        labels.append(row["cancer"])
    return images, labels


class CancerImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        label = int(self.df.loc[idx, "cancer"])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def get_dataloaders(
    train_df,
    test_df,
    root_dir,
    batch_size=16,
    config_path="config/config.yaml",
    img_size=None,
):
    import cv2  # Bổ sung import cv2 cho augment dùng interpolation

    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    # img_size phải là tuple (H, W)
    # Đảm bảo img_size là tuple (height, width)
    if isinstance(img_size, (list, tuple)):
        if len(img_size) == 2:
            height, width = int(img_size[0]), int(img_size[1])
        else:
            height = width = int(img_size[0])
    else:
        height = width = int(img_size)
    # print(f"[get_dataloaders] Resize to (height, width): ({height}, {width})")

    # Đảm bảo height luôn là chiều lớn hơn width nếu bạn muốn resize về hình ngang
    # Nếu muốn resize về hình vuông, ép height = width = max(height, width)
    # Ví dụ ép về hình vuông lớn nhất:
    # max_side = max(height, width)
    # height, width = max_side, max_side

    train_transform = A.Compose(
        [
            # Đảm bảo Resize là augmentation cuối cùng để giữ kích thước đồng nhất
            A.Resize(height, width),
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
            # Sử dụng đúng tham số cho CoarseDropout (albumentations >=1.3.0)
            # Chỉ dùng max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value, mask_fill_value, p
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
            A.CLAHE(
                clip_limit=2.0, tile_grid_size=(8, 8), p=0.3
            ),  # Increased for mammograms
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),  # Not suitable for grayscale mammograms
            # A.ToGray(p=0.2),  # Mammograms are already grayscale
            A.Equalize(p=0.3),  # Increased probability - very useful for mammograms
            # A.HEStain(p=0.2),  # Only for histopathology, not X-ray images
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),  # Modified for mammograms
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),  # Not suitable for grayscale
            # Additional augmentations good for mammograms:
            A.Sharpen(
                alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2
            ),  # Enhance edges/calcifications
            A.UnsharpMask(
                blur_limit=(3, 5), sigma_limit=(1.0, 2.0), alpha=(0.1, 0.3), p=0.2
            ),  # Detail enhancement
            A.GaussNoise(p=0.1),
            A.Resize(height, width),
            A.Normalize([0.5] * 3, [0.5] * 3),
            ToTensorV2(),
            # Đặt Resize cuối cùng nếu bạn có crop/pad ở trên (nếu không thì giữ nguyên)
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(height, width),
            A.Normalize([0.5] * 3, [0.5] * 3),
            ToTensorV2(),
        ]
    )
    train_dataset = CancerImageDataset(train_df, root_dir, train_transform)
    test_dataset = CancerImageDataset(test_df, root_dir, test_transform)
    # WeightedRandomSampler cho train_loader
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
        num_workers=4,  # Thử với 4 workers
        pin_memory=True,  # Tăng tốc chuyển dữ liệu sang GPU
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Thử với 4 workers
        pin_memory=True,  # Tăng tốc chuyển dữ liệu sang GPU
    )
    return train_loader, test_loader
