import os
import torch
from torch import nn
import pandas as pd
import argparse

from src.gradcam.gradcam_utils_based import pre_gradcam
from src.gradcam.gradcam_utils_patch import pre_mil_gradcam
from src.gradcam.gradcam_utils_patch import split_image_into_patches


def load_data_bbx3(data_folder):
    metadata_path = os.path.join(data_folder, "metadata2.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    # Create bbx column from x, y, width, height columns
    df["bbx"] = df[["x", "y", "width", "height"]].apply(
        lambda row: [row["x"], row["y"], row["width"], row["height"]], axis=1
    )
    # Group all bbx by image_id into a list
    bbx_grouped = (
        df.groupby("image_id")["bbx"]
        .apply(list)
        .reset_index()
        .rename(columns={"bbx": "bbx_list"})
    )
    df = df.merge(bbx_grouped, on="image_id", how="left")
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    return train_df, test_df


def load_full_model(
    model_path: str,
) -> tuple[
    nn.Module,
    tuple[int, int],
    str | None,
    str | None,
    dict[str, list[float]] | None,
    int | None,
    str | None,
]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    model.eval()

    input_size = checkpoint.get("input_size", (448, 448))
    model_name = checkpoint.get("model_name", None)
    gradcam_layer = checkpoint.get("gradcam_layer", None)
    normalize = checkpoint.get("normalize", None)
    num_patches = checkpoint.get("num_patches", None)
    arch_type = checkpoint.get("arch_type", None)

    return (
        model,
        input_size,
        model_name,
        gradcam_layer,
        normalize,
        num_patches,
        arch_type,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument(
        "--random", type=int, default=42, help="Sample index to use (default: 42)"
    )
    args = parser.parse_args()

    # Load model and metadata
    model_tuple = load_full_model(args.pretrained_model_path)
    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        model_tuple
    )

    # Get test_df from load_data_bbx3
    _, test_df = load_data_bbx3(args.dataset_folder)

    # Select sample index from --random argument
    sample_idx = args.random
    if sample_idx >= len(test_df):
        sample_idx = 0  # fallback to first if out of range

    # Get image path, bounding box, ground truth from test_df
    image_path = os.path.join(args.dataset_folder, test_df.iloc[sample_idx]["link"])
    bbx_list = (
        test_df.iloc[sample_idx]["bbx_list"] if "bbx_list" in test_df.columns else None
    )
    gt = test_df.iloc[sample_idx]["cancer"] if "cancer" in test_df.columns else None

    # Prepare input for GradCAM
    if arch_type == "based":
        # Standard model
        result = pre_gradcam(model_tuple[:5], image_path)
    else:
        # Patch/MIL model
        result = pre_mil_gradcam(model_tuple, image_path)

    # Unpack result
    model_out, input_tensor, img, target_layer, class_idx, pred_class, prob_class = (
        result
    )

    print("Model:", model_out)
    print("Input tensor shape:", input_tensor.shape)
    print("Image:", img)
    print("Target layer:", target_layer)
    print("Class idx (input):", class_idx)
    print("Predicted class:", pred_class)
    print("Predicted prob:", prob_class)
    print("Bounding box list:", bbx_list)
    print("Ground truth:", gt)


if __name__ == "__main__":
    main()
