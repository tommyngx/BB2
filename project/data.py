import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def load_data(data_folder):
    metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df = df.dropna()
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
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
    img_size = get_image_size_from_config(config_path)
    train_transform = A.Compose(
        [
            A.Resize(*img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
