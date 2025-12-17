"""
Preprocessing utilities for DETR detection
Handles grouping multiple bboxes per image_id with robust validation
"""

import pandas as pd
import numpy as np
import cv2
import os


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


def group_bboxes_by_image(df, data_folder, validate_bbox=True, min_area=100):
    """
    Group multiple bbox annotations by image_id with image-based validation
    OPTIMIZED: Batch process image dimensions, vectorized bbox validation
    """
    # Pre-load image dimensions for all unique images (parallel if needed)
    print("  Loading image dimensions...")
    unique_links = df[["image_id", "link"]].drop_duplicates("image_id")

    img_dims = {}
    skipped_images = 0

    for _, row in unique_links.iterrows():
        img_path = os.path.join(data_folder, row["link"])
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img_dims[row["image_id"]] = (img.shape[0], img.shape[1])  # (h, w)
            else:
                skipped_images += 1
        except:
            skipped_images += 1

    if skipped_images > 0:
        print(f"⚠️ Skipped {skipped_images} images due to read errors")

    # Vectorized bbox validation
    def validate_bboxes_batch(group, img_h, img_w):
        """Validate all bboxes for one image at once"""
        # Get bbox columns
        bbox_cols = ["x", "y", "width", "height"]
        if not all(col in group.columns for col in bbox_cols):
            return []

        # Filter valid rows
        valid_rows = group[group[bbox_cols].notna().all(axis=1)].copy()
        if len(valid_rows) == 0:
            return []

        # Vectorized operations
        bboxes = valid_rows[bbox_cols].values.astype(float)

        # Clip to boundaries
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img_w - 1)  # x
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img_h - 1)  # y
        bboxes[:, 2] = np.maximum(0, bboxes[:, 2])  # w
        bboxes[:, 3] = np.maximum(0, bboxes[:, 3])  # h

        # Ensure bbox doesn't exceed bounds
        bboxes[:, 2] = np.minimum(bboxes[:, 2], img_w - bboxes[:, 0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], img_h - bboxes[:, 1])

        # Filter by area
        areas = bboxes[:, 2] * bboxes[:, 3]
        valid_mask = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0) & (areas >= min_area)

        return bboxes[valid_mask].tolist()

    # Group and process
    grouped_rows = []
    for image_id, group in df.groupby("image_id"):
        first_row = group.iloc[0]

        # Skip if image dimensions not available
        if image_id not in img_dims:
            continue

        img_h, img_w = img_dims[image_id]

        # Validate bboxes in batch
        if validate_bbox:
            bbox_list = validate_bboxes_batch(group, img_h, img_w)
        else:
            # No validation, just extract
            bbox_cols = ["x", "y", "width", "height"]
            valid_rows = group[group[bbox_cols].notna().all(axis=1)]
            bbox_list = valid_rows[bbox_cols].values.tolist()

        # Create row
        new_row = {
            "image_id": image_id,
            "link": first_row["link"],
            "cancer": int(first_row["cancer"]),
            "split": first_row["split"],
            "bbox_list": bbox_list,
            "num_bboxes": len(bbox_list),
            "image_width": img_w,
            "image_height": img_h,
        }

        # Copy metadata
        for col in [
            "original_width",
            "original_height",
            "patient_id",
            "laterality",
            "view",
        ]:
            if col in first_row.index and pd.notna(first_row[col]):
                new_row[col] = first_row[col]

        grouped_rows.append(new_row)

    return pd.DataFrame(grouped_rows)


def prepare_detr_dataframe(
    df, data_folder, validate_bbox=True, min_area=100, verbose=True
):
    """
    Main preprocessing function for DETR detection
    Converts multi-row bbox annotations to single-row with bbox_list

    Args:
        df: Raw DataFrame from metadata.csv (can have duplicate image_ids)
        data_folder: Root folder containing images
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area threshold
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df) with grouped bboxes
        Each DataFrame has one row per image with bbox_list column
    """
    # Group bboxes by image_id with validation
    df_grouped = group_bboxes_by_image(
        df, data_folder, validate_bbox=validate_bbox, min_area=min_area
    )

    # Split train/test
    train_df = df_grouped[df_grouped["split"] == "train"].copy().reset_index(drop=True)
    test_df = df_grouped[df_grouped["split"] == "test"].copy().reset_index(drop=True)

    if verbose:
        print(f"\n📊 DETR Preprocessing Results:")
        print(f"  Original annotations: {len(df)} rows")
        print(f"  Grouped images: {len(df_grouped)} unique images")
        print(f"    Train: {len(train_df)} images")
        print(f"    Test:  {len(test_df)} images")

        # Multi-bbox statistics
        train_multi = (train_df["num_bboxes"] > 1).sum()
        test_multi = (test_df["num_bboxes"] > 1).sum()
        print(f"\n  Images with >1 bbox:")
        print(
            f"    Train: {train_multi}/{len(train_df)} ({train_multi / len(train_df) * 100:.1f}%)"
        )
        print(
            f"    Test:  {test_multi}/{len(test_df)} ({test_multi / len(test_df) * 100:.1f}%)"
        )

        if train_multi > 0:
            print(f"  Max bboxes per image (train): {train_df['num_bboxes'].max()}")
        if test_multi > 0:
            print(f"  Max bboxes per image (test):  {test_df['num_bboxes'].max()}")

        # Label distribution
        print(f"\n  Label distribution:")
        train_counts = train_df["cancer"].value_counts().sort_index()
        test_counts = test_df["cancer"].value_counts().sort_index()
        all_labels = sorted(set(train_counts.index).union(set(test_counts.index)))
        print("  Label | Train  | Test")
        for label in all_labels:
            n_train = train_counts.get(label, 0)
            n_test = test_counts.get(label, 0)
            print(f"    {label}   | {n_train:5d}  | {n_test:5d}")

        print("=" * 60)

    return train_df, test_df


def prepare_detr_from_metadata(
    metadata_path, data_folder, validate_bbox=True, min_area=100, verbose=True
):
    """
    Load metadata.csv and prepare DETR-ready DataFrames

    Args:
        metadata_path: Path to metadata.csv
        data_folder: Root folder containing images (needed for validation)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df, df_raw)
        - train_df, test_df: Grouped DataFrames ready for DETR
        - df_raw: Original raw DataFrame
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load raw metadata
    df_raw = pd.read_csv(metadata_path)

    if verbose:
        print(f"📂 Loading: {metadata_path}")
        print(f"  Raw rows: {len(df_raw)}")
        print(f"  Unique images: {df_raw['image_id'].nunique()}")

    # Prepare DETR dataframes
    train_df, test_df = prepare_detr_dataframe(
        df_raw,
        data_folder,
        validate_bbox=validate_bbox,
        min_area=min_area,
        verbose=verbose,
    )

    return train_df, test_df, df_raw


# ============= Example Usage =============
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.data.m2_data_preprocess <metadata.csv> <data_folder>"
        )
        sys.exit(1)

    metadata_path = sys.argv[1]
    data_folder = sys.argv[2]

    # Process metadata
    train_df, test_df, df_raw = prepare_detr_from_metadata(
        metadata_path, data_folder, validate_bbox=True, min_area=100, verbose=True
    )

    # Show sample
    print("\n📋 Sample from train_df:")
    print(train_df.head(3))

    # Show multi-bbox examples
    multi_bbox_examples = train_df[train_df["num_bboxes"] > 1].head(3)
    if len(multi_bbox_examples) > 0:
        print("\n📋 Examples with >1 bbox:")
        for _, row in multi_bbox_examples.iterrows():
            print(f"  image_id: {row['image_id']}")
            print(f"    bboxes: {row['bbox_list']}")
