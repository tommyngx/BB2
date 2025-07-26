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
    df = df.dropna()
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
    train_df, test_df, root_dir, batch_size=16, config_path="config/config.yaml"
):
    import cv2  # Bổ sung import cv2 cho augment dùng interpolation

    img_size = get_image_size_from_config(config_path)
    train_transform = A.Compose(
        [
            A.Resize(*img_size),
            A.OneOf(
                [
                    # Sử dụng đúng tham số cho Downscale (albumentations >=1.3.0)
                    # Downscale chỉ nhận scale và interpolation (không nhận scale_min/scale_max)
                    # Nếu muốn random scale, dùng A.Downscale(scale=0.75, interpolation=cv2.INTER_LINEAR, p=0.1)
                    # hoặc dùng A.Downscale(scale=0.95, interpolation=cv2.INTER_LINEAR, p=0.1)
                    # Nếu muốn random hóa, bạn cần tự chọn 1 giá trị scale ngẫu nhiên trước khi truyền vào Compose.
                    A.Downscale(scale=0.75, interpolation=cv2.INTER_LINEAR, p=0.1),
                    A.Downscale(scale=0.75, interpolation=cv2.INTER_LANCZOS4, p=0.1),
                    A.Downscale(scale=0.95, interpolation=cv2.INTER_LINEAR, p=0.8),
                ],
                p=0.125,
            ),
            # Sử dụng đúng tham số cho CoarseDropout (albumentations >=1.3.0)
            # Chỉ dùng max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value, mask_fill_value, p
            A.CoarseDropout(
                max_holes=3,
                max_height=int(img_size[0] * 0.08),
                max_width=int(img_size[1] * 0.15),
                min_holes=1,
                min_height=1,
                min_width=1,
                fill_value=0,
                mask_fill_value=None,
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
            A.ElasticTransform(
                alpha=1,
                sigma=20,
                p=0.1,
            ),
            # A.RandomCrop(height=img_size, width=img_size, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.1,
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2
            ),
            A.GaussNoise(p=0.1),
            A.Normalize([0.5] * 3, [0.5] * 3),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [A.Resize(*img_size), A.Normalize([0.5] * 3, [0.5] * 3), ToTensorV2()]
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
