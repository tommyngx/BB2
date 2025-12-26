import os
import pandas as pd
import numpy as np


def validate_bboxes_vectorized(df, min_area=100):
    """
    Validate and clip bboxes using vectorized pandas operations
    NO image loading - uses img_width, img_height from CSV
    """
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


def get_image_dimensions(df, data_folder):
    """
    Get image dimensions from CSV or by loading first occurrence of each image
    """
    import cv2

    if "img_width" in df.columns and "img_height" in df.columns:
        return {}
    unique_links = df[["image_id", "link"]].drop_duplicates("image_id")
    img_dims = {}
    for _, row in unique_links.iterrows():
        img_path = os.path.join(data_folder, row["link"])
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img_dims[row["image_id"]] = (img.shape[0], img.shape[1])
        except:
            pass
    return img_dims


def remove_large_bboxes(df, area_ratio_thresh=0.9):
    """Remove bboxes that occupy more than area_ratio_thresh of image area"""
    df = df.copy()
    df["bbox_area"] = df["width"] * df["height"]
    df["img_area"] = df["img_width"] * df["img_height"]
    df["bbox_ratio"] = df["bbox_area"] / df["img_area"]
    df = df[df["bbox_ratio"] <= area_ratio_thresh].copy()
    df = df.drop(columns=["bbox_area", "img_area", "bbox_ratio"])
    return df


def unique_patients(df, debug_save_path=None):
    """
    Count unique patients (normalized) from DataFrame
    Ensures drop_duplicates by image_id/link before counting
    """

    def find_col(df, candidates):
        df_cols = list(df.columns)
        for col in candidates:
            for c in df_cols:
                if c.lower() == col:
                    return c
        return None

    patient_id_col = find_col(df, ["patient_id", "id", "image_id"])
    if patient_id_col is None:
        return 0

    if "link" in df.columns:
        df_unique = df.drop_duplicates(subset=["link"])
    elif "image_id" in df.columns:
        df_unique = df.drop_duplicates(subset=["image_id"])
    else:
        df_unique = df

    def normalize_pid(val):
        if isinstance(val, str):
            for suffix in ["_R", "_L", "_MLO", "_CC"]:
                val = val.split(suffix)[0]
            return val
        return val

    norm_pid_series = df_unique[patient_id_col].map(normalize_pid)
    if debug_save_path is not None:
        debug_df = df_unique.copy()
        debug_df["normalized_patient_id"] = norm_pid_series
        debug_df.to_csv(debug_save_path, index=False)
        print(f"⚡️ Saved unique patient debug DataFrame to: {debug_save_path}")
    return norm_pid_series.nunique()


def group_bboxes_by_image_vectorized(
    df, data_folder=None, validate_bbox=True, min_area=100
):
    """
    Group multiple bbox annotations by image_id using VECTORIZED operations
    Keep ALL images, even those without valid bboxes
    """
    if "img_width" not in df.columns or "img_height" not in df.columns:
        print("  ⚠️ Missing img_width/img_height in CSV, loading from images...")
        if data_folder is None:
            raise ValueError(
                "data_folder required when img_width/img_height not in CSV"
            )
        img_dims = get_image_dimensions(df, data_folder)
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
    Main preprocessing function
    OPTIMIZED: Uses vectorized pandas operations
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

        # Unique patients statistics
        n_train_patients = unique_patients(train_df)
        n_test_patients = unique_patients(test_df)
        combined_df = pd.concat([train_df, test_df])
        n_total_patients = unique_patients(combined_df)
        pct_train = (
            (n_train_patients / n_total_patients * 100) if n_total_patients > 0 else 0
        )
        pct_test = (
            (n_test_patients / n_total_patients * 100) if n_total_patients > 0 else 0
        )
        print(
            f"    Unique patients: train {n_train_patients} ({pct_train:.1f}%) | "
            f"test {n_test_patients} ({pct_test:.1f}%) | total {n_total_patients}"
        )

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
        train_total = len(train_df)
        test_total = len(test_df)
        print("  Label | Train  | %Train  | Test   | %Test   | Total  | %Total")
        for label in all_labels:
            n_train = train_counts.get(label, 0)
            n_test = test_counts.get(label, 0)
            n_total = n_train + n_test
            total = train_total + test_total
            pct_train = (n_train / train_total * 100) if train_total > 0 else 0
            pct_test = (n_test / test_total * 100) if test_total > 0 else 0
            pct_total = (n_total / total * 100) if total > 0 else 0
            print(
                f"    {label}   | {n_train:5d}  | {pct_train:6.2f}% | "
                f"{n_test:5d}  | {pct_test:6.2f}% | {n_total:5d}  | {pct_total:6.2f}%"
            )
        print("=" * 60)

        # Thêm thống kê cơ bản từ print_dataset_stats2
        try:
            from .dataloader import print_dataset_stats2

            print_dataset_stats2(train_df, test_df, name="Detection (grouped)")
        except Exception as e:
            print(f"[WARN] Could not print print_dataset_stats2: {e}")

    return train_df, test_df


def prepare_from_metadata(
    metadata_path, data_folder=None, validate_bbox=True, min_area=100, verbose=True
):
    """
    Load metadata.csv and prepare detection-ready DataFrames
    OPTIMIZED: No image loading if img_width/img_height in CSV
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
