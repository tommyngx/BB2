import os
import pandas as pd
import numpy as np


def validate_bboxes_vectorized(df, min_area=100):
    df = df.copy()
    bbox_cols = ["x", "y", "width", "height"]
    for col in bbox_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].clip(lower=0.0)
    df["x"] = df["x"].clip(upper=df["img_width"] - 1)
    df["y"] = df["y"].clip(upper=df["img_height"] - 1)
    df["width"] = np.minimum(df["width"], df["img_width"] - df["x"])
    df["height"] = np.minimum(df["height"], df["img_height"] - df["y"])
    df["bbox_area"] = df["width"] * df["height"]
    df["is_valid_bbox"] = (
        (df["width"] > 0) & (df["height"] > 0) & (df["bbox_area"] >= min_area)
    )
    return df


def remove_large_bboxes(df, area_ratio_thresh=0.9):
    df = df.copy()
    df["bbox_area"] = df["width"] * df["height"]
    df["img_area"] = df["img_width"] * df["img_height"]
    df["bbox_ratio"] = df["bbox_area"] / df["img_area"]
    df = df[df["bbox_ratio"] <= area_ratio_thresh].copy()
    df = df.drop(columns=["bbox_area", "img_area", "bbox_ratio"])
    return df


def group_bboxes_by_image_vectorized(
    df, data_folder=None, validate_bbox=True, min_area=100
):
    if "img_width" not in df.columns or "img_height" not in df.columns:
        if data_folder is None:
            raise ValueError(
                "data_folder required when img_width/img_height not in CSV"
            )
        import cv2

        unique_links = df[["image_id", "link"]].drop_duplicates("image_id")
        img_dims = {}
        for _, row in unique_links.iterrows():
            img_path = os.path.join(data_folder, row["link"])
            img = cv2.imread(img_path)
            if img is not None:
                img_dims[row["image_id"]] = (img.shape[0], img.shape[1])
        df["img_height"] = df["image_id"].map(lambda x: img_dims.get(x, (0, 0))[0])
        df["img_width"] = df["image_id"].map(lambda x: img_dims.get(x, (0, 0))[1])
        df = df[(df["img_height"] > 0) & (df["img_width"] > 0)].copy()
    if validate_bbox:
        df = validate_bboxes_vectorized(df, min_area=min_area)
    else:
        df["is_valid_bbox"] = True

    def aggregate_bboxes(group):
        bbox_cols = ["x", "y", "width", "height"]
        if all(col in group.columns for col in bbox_cols):
            valid_rows = (
                group[group["is_valid_bbox"]]
                if "is_valid_bbox" in group.columns
                else group
            )
            if len(valid_rows) > 0:
                bboxes = valid_rows[bbox_cols].values.tolist()
                bboxes = [
                    bbox
                    for bbox in bboxes
                    if not all(v == 0 or pd.isna(v) for v in bbox)
                ]
            else:
                bboxes = []
        else:
            bboxes = []
        return bboxes

    grouped = (
        df.groupby("image_id")
        .agg(
            {
                "link": "first",
                "cancer": "first",
                "split": "first",
                "img_width": "first",
                "img_height": "first",
            }
        )
        .reset_index()
    )
    grouped["bbox_list"] = df.groupby("image_id").apply(aggregate_bboxes).values
    grouped["num_bboxes"] = grouped["bbox_list"].apply(len)
    metadata_cols = [
        "original_width",
        "original_height",
        "patient_id",
        "laterality",
        "view",
    ]
    for col in metadata_cols:
        if col in df.columns:
            grouped[col] = df.groupby("image_id")[col].first().values
    return grouped


def prepare_dataframe(
    df, data_folder=None, validate_bbox=True, min_area=100, verbose=True
):
    """
    Main preprocessing function for detection
    OPTIMIZED: Uses vectorized pandas operations

    Args:
        df: Raw DataFrame from metadata.csv
        data_folder: Root folder (optional if img_width/img_height in CSV)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df) with grouped bboxes
    """
    if verbose:
        print(f"\n📊 Preprocessing (Vectorized):")
        print(f"  Input: {len(df)} annotations")
    df = remove_large_bboxes(df, area_ratio_thresh=0.9)
    df_grouped = group_bboxes_by_image_vectorized(
        df, data_folder=data_folder, validate_bbox=validate_bbox, min_area=min_area
    )
    train_df = df_grouped[df_grouped["split"] == "train"].copy().reset_index(drop=True)
    test_df = df_grouped[df_grouped["split"] == "test"].copy().reset_index(drop=True)
    if verbose:
        print(f"  Output: {len(df_grouped)} unique images")
        print(f"    Train: {len(train_df)} images")
        print(f"    Test:  {len(test_df)} images")
    return train_df, test_df


def prepare_from_metadata(
    metadata_path, data_folder=None, validate_bbox=True, min_area=100, verbose=True
):
    """
    Load metadata.csv and prepare detection-ready DataFrames
    OPTIMIZED: No image loading if img_width/img_height in CSV

    Args:
        metadata_path: Path to metadata.csv
        data_folder: Root folder (optional if dimensions in CSV)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df, df_raw)
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    df_raw = pd.read_csv(metadata_path)
    if verbose:
        print(f"📂 Loading: {metadata_path}")
        print(f"  Raw rows: {len(df_raw)}")
        print(f"  Unique images: {df_raw['image_id'].nunique()}")
        has_dims = "img_width" in df_raw.columns and "img_height" in df_raw.columns
        if has_dims:
            print(f"  ✅ Image dimensions found in CSV (fast mode)")
        else:
            print(f"  ⚠️ Image dimensions NOT in CSV, will load images (slow)")
    train_df, test_df = prepare_dataframe(
        df_raw,
        data_folder=data_folder,
        validate_bbox=validate_bbox,
        min_area=min_area,
        verbose=verbose,
    )
    return train_df, test_df, df_raw
