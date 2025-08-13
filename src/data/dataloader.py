import os
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import WeightedRandomSampler


def get_image_size_from_config(config_path="src/config/config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    img_size = config.get("image_size", 448)
    if isinstance(img_size, (list, tuple)):
        return tuple(img_size)
    return (img_size, img_size)


def get_target_column_from_config(config_path="src/config/config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("target_column", "cancer")


def get_num_patches_from_config(config_path="src/config/config.yaml", num_patches=None):
    if num_patches is not None:
        return num_patches
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("num_patches", 2)


def get_model_key(
    config_path="config/config.yaml", base_model_key="dataset", num_patches=None
):
    num_patches = get_num_patches_from_config(config_path, num_patches)
    patch_suffix = f"p{num_patches}"
    return f"{base_model_key}_{patch_suffix}"


def load_metadata(data_folder, config_path="src/config/config.yaml"):
    target_column = get_target_column_from_config(config_path)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df = df.drop_duplicates(subset=["link"])
    if not np.issubdtype(df[target_column].dtype, np.number):
        df["target_label"], class_names = pd.factorize(df[target_column])
        label_col = "target_label"
    else:
        label_col = target_column
        class_names = sorted(df[target_column].unique())
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    train_df["cancer"] = train_df[label_col]
    test_df["cancer"] = test_df[label_col]
    return train_df, test_df, class_names


def get_weighted_sampler(train_df, label_col="cancer"):
    class_counts = train_df[label_col].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[train_df.iloc[i][label_col]] for i in range(total_samples)]
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=total_samples, replacement=True
    )
    return sampler
