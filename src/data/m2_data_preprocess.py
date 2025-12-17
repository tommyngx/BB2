"""
Preprocessing utilities for DETR detection
Handles grouping multiple bboxes per image_id
"""

import pandas as pd
import numpy as np
import json


def group_bboxes_by_image(df, validate_bbox=True, min_area=100):
    """
    Group multiple bbox annotations by image_id

    Args:
        df: DataFrame with columns [image_id, link, cancer, x, y, width, height, split, ...]
            Can have multiple rows per image_id (one row per bbox)
        validate_bbox: Whether to validate and filter invalid bboxes
        min_area: Minimum bbox area (pixels) to be considered valid

    Returns:
        DataFrame with one row per image_id, with bbox_list column containing all bboxes
        Columns: [image_id, link, cancer, split, bbox_list, num_bboxes, ...]
    """
    grouped_rows = []

    for image_id, group in df.groupby("image_id"):
        # Get first row for image-level metadata
        first_row = group.iloc[0]

        # Collect all valid bboxes for this image
        bbox_list = []

        for _, row in group.iterrows():
            # Check if bbox columns exist and are valid
            if all(col in row.index for col in ["x", "y", "width", "height"]):
                if all(pd.notna(row[col]) for col in ["x", "y", "width", "height"]):
                    x, y, w, h = row["x"], row["y"], row["width"], row["height"]

                    if validate_bbox:
                        # Ensure non-negative
                        x = max(0, float(x))
                        y = max(0, float(y))
                        w = max(0, float(w))
                        h = max(0, float(h))

                        # Validate positive area and min size
                        if w > 0 and h > 0 and (w * h) >= min_area:
                            bbox_list.append([x, y, w, h])
                    else:
                        # No validation, just convert to list
                        bbox_list.append([float(x), float(y), float(w), float(h)])

        # Create new row with grouped bboxes
        new_row = {
            "image_id": image_id,
            "link": first_row["link"],
            "cancer": int(first_row["cancer"]),
            "split": first_row["split"],
            "bbox_list": bbox_list,  # List of lists: [[x,y,w,h], ...]
            "num_bboxes": len(bbox_list),
        }

        # Copy other metadata columns if exist
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

    # Create grouped DataFrame
    df_grouped = pd.DataFrame(grouped_rows)

    return df_grouped


def prepare_detr_dataframe(df, validate_bbox=True, min_area=100, verbose=True):
    """
    Main preprocessing function for DETR detection
    Converts multi-row bbox annotations to single-row with bbox_list

    Args:
        df: Raw DataFrame from metadata.csv (can have duplicate image_ids)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area threshold
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df) with grouped bboxes
        Each DataFrame has one row per image with bbox_list column
    """
    # Group bboxes by image_id
    df_grouped = group_bboxes_by_image(
        df, validate_bbox=validate_bbox, min_area=min_area
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
    metadata_path, validate_bbox=True, min_area=100, verbose=True
):
    """
    Load metadata.csv and prepare DETR-ready DataFrames

    Args:
        metadata_path: Path to metadata.csv
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df, df_raw)
        - train_df, test_df: Grouped DataFrames ready for DETR
        - df_raw: Original raw DataFrame
    """
    import os

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
        df_raw, validate_bbox=validate_bbox, min_area=min_area, verbose=verbose
    )

    return train_df, test_df, df_raw


# ============= Example Usage =============
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.m2_data_preprocess <metadata.csv>")
        sys.exit(1)

    metadata_path = sys.argv[1]

    # Process metadata
    train_df, test_df, df_raw = prepare_detr_from_metadata(
        metadata_path, validate_bbox=True, min_area=100, verbose=True
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
