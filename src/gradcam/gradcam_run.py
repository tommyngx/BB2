import os
import torch
from torch import nn
import pandas as pd
import argparse
from PIL import Image
import numpy as np

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


def print_gradcam_info(
    model_out,
    input_tensor,
    img,
    target_layer,
    class_idx,
    pred_class,
    prob_class,
    bbx_list,
    gt,
):
    print("Model name:", type(model_out).__name__)
    print("Input tensor shape:", input_tensor.shape)
    print("Image:", img)
    print("Target layer:", target_layer)
    print("Class idx (input):", class_idx)
    print("Predicted class:", pred_class)
    print("Predicted prob:", prob_class)
    print("Bounding box list:", bbx_list)
    print("Ground truth:", gt)


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
        (
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
        ) = pre_gradcam(model_tuple[:5], image_path, target_layer=None, class_idx=None)
        print_gradcam_info(
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
            bbx_list,
            gt,
        )
        from src.gradcam.gradcam_utils_based import (
            gradcam,
            gradcam_plus_plus,
            post_gradcam,
        )

        gradcam_map = gradcam(model_out, input_tensor, target_layer, class_idx)
        # gradcam_map = gradcam_plus_plus(model_out, input_tensor, target_layer, class_idx)
        post_gradcam(
            gradcam_map,
            img,
            bbx_list=bbx_list,
            option=5,
            blend_alpha=0.5,
            pred=pred_class,
            prob=prob_class,
            gt_label=gt,
        )
    else:
        # Patch/MIL model
        result = pre_mil_gradcam(model_tuple, image_path)
        (
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
        ) = result
        print_gradcam_info(
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
            bbx_list,
            gt,
        )

        from src.gradcam.gradcam_utils_patch import (
            mil_gradcam,
            mil_gradcam_plus_plus,
            post_mil_gradcam,
        )

        gradcam_map = mil_gradcam(model_out, input_tensor, target_layer, class_idx)

        print(f"\nGradCAM shape: {gradcam_map.shape}, dtype: {gradcam_map.dtype}")

        # Visualize each patch with its own GradCAM
        if gradcam_map.ndim == 3:
            num_patches_result = gradcam_map.shape[0]

            # Get model metadata
            _, input_size_meta, _, _, _, num_patches_meta, arch_type_meta = model_tuple

            # Check if has global image
            has_global = (
                "v4" in arch_type_meta.lower() or "global" in arch_type_meta.lower()
            )

            # Create patch images for visualization
            # Need to recreate the EXACT same patches as in pre_mil_gradcam
            patch_images = split_image_into_patches(
                img, num_patches_meta, input_size_meta, add_global=True
            )

            print(
                f"\nVisualizing {num_patches_result} patches{' (including 1 global image)' if has_global else ''}:"
            )
            print(f"  Created {len(patch_images)} patch images for visualization")

            # Verify consistency
            if num_patches_result != len(patch_images):
                print(
                    f"  ⚠️ WARNING: GradCAM has {num_patches_result} heatmaps but we have {len(patch_images)} patch images!"
                )
                print(
                    f"     Using min({num_patches_result}, {len(patch_images)}) for visualization"
                )
                num_to_visualize = min(num_patches_result, len(patch_images))
            else:
                num_to_visualize = num_patches_result

            # Visualize each patch with its heatmap
            for patch_idx in range(num_to_visualize):
                patch_cam = gradcam_map[patch_idx]  # Shape: (H, W)
                patch_img = patch_images[patch_idx]  # PIL Image or np.ndarray

                # Convert to PIL if needed
                if isinstance(patch_img, np.ndarray):
                    patch_img = Image.fromarray(patch_img)

                # Determine if this is the global patch
                is_global = has_global and (patch_idx == num_to_visualize - 1)

                print(
                    f"\n=== {'Global Image' if is_global else f'Patch {patch_idx + 1}/{num_patches_meta}'} ==="
                )
                print(f"  Patch image size: {patch_img.size} (W×H)")
                print(f"  Heatmap shape: {patch_cam.shape} (H×W)")

                # Create custom pred string
                if is_global:
                    pred_str = f"Global: {pred_class}"
                else:
                    pred_str = f"Patch {patch_idx + 1}: {pred_class}"

                # Visualize
                post_mil_gradcam(
                    patch_cam,
                    patch_img,
                    bbx_list=None,
                    option=3,
                    blend_alpha=0.5,
                    pred=pred_str,
                    prob=prob_class,
                    gt_label=None,
                )
        else:
            # Standard model - single heatmap
            post_mil_gradcam(
                gradcam_map,
                img,
                bbx_list=bbx_list,
                option=5,
                blend_alpha=0.5,
                pred=pred_class,
                prob=prob_class,
                gt_label=gt,
            )


if __name__ == "__main__":
    main()
