import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .augment import get_train_augmentation, get_test_augmentation
from .dataloader import get_image_size_from_config, get_weighted_sampler


class CancerImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        if root_dir is None:
            raise ValueError(
                "root_dir must not be None. Please provide the image folder path."
            )
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
    num_workers=4,
    pin_memory=True,
):
    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    if isinstance(img_size, (list, tuple)):
        height, width = int(img_size[0]), int(img_size[1])
    else:
        height = width = int(img_size)
    train_transform = get_train_augmentation(height, width, resize_first=True)
    test_transform = get_test_augmentation(height, width)
    train_dataset = CancerImageDataset(train_df, root_dir, train_transform)
    test_dataset = CancerImageDataset(test_df, root_dir, test_transform)
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
