"""
Testing script for patch-based/MIL classification models with GradCAM visualization
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src2.data.patch_dataset import get_dataloaders
from src2.data.dataloader import load_metadata
from src2.models.patch_model import get_patch_model
from src2.utils.common import load_config, get_arg_or_config
from src2.utils.gradcam_utils import get_gradcam_layer
from src2.utils.based_viz_utils import visualize_classification_result
from src2.utils.based_test_utils import (
    evaluate_classification_model,
    compute_classification_metrics,
    print_test_metrics,
)

# Import MIL GradCAM functions
from src2.gradcam.gradcam_utils_patch import mil_gradcam, split_image_into_patches


def parse_img_size(val):
    if val is None:
        return None
    if isinstance(val, str) and "x" in val:
        h, w = val.lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def load_image_metadata(data_folder, test_df):
    """Load image metadata including original size and path"""
    image_info = {}

    for idx, row in test_df.iterrows():
        image_id = str(row["image_id"])
        image_path = os.path.join(data_folder, row["link"])

        # Get original size
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                original_size = (img.height, img.width)
        except:
            original_size = (448, 448)  # fallback

        # Get bboxes
        gt_bboxes = row.get("bbox_list", None)

        image_info[image_id] = {
            "original_size": original_size,
            "image_path": image_path,
            "bbx_list": gt_bboxes,
        }

    return image_info


def get_gradcam_layer_patch(model, model_type, arch_type):
    """
    Return the name of the appropriate layer for GradCAM based on MIL/patch architecture.
    """
    # MIL architectures (mil, mil_v2, mil_v3, ..., mil_v12)
    if "mil" in arch_type:
        # For MIL models, check multiple possible layer names
        if hasattr(model, "base_model"):
            if hasattr(model.base_model, "layer4"):
                return "base_model.layer4"
            elif hasattr(model.base_model, "stages"):
                return "base_model.stages.3"
            elif hasattr(model.base_model, "blocks"):
                return "base_model.blocks.5"
        return "base_model.layer4"

    # Patch ResNet
    elif arch_type == "patch_resnet":
        return "base_model.layer4"

    # Patch Transformer
    elif arch_type == "patch_transformer":
        if hasattr(model, "transformer_encoder"):
            return "transformer_encoder.layers.-1"
        return "transformer_encoder"

    # Token Mixer
    elif arch_type == "token_mixer":
        if hasattr(model, "token_mixer"):
            return "token_mixer"
        return "feature_extractor.layer4"

    # Global-Local architectures
    elif "global_local" in arch_type:
        if hasattr(model, "local_feature_extractor"):
            return "local_feature_extractor.layer4"
        elif hasattr(model, "feature_extractor"):
            return "feature_extractor.layer4"
        return "feature_extractor"

    # Fallback
    else:
        if hasattr(model, "feature_extractor"):
            fe_children = list(model.feature_extractor.named_children())
            if fe_children:
                return f"feature_extractor.{fe_children[-1][0]}"

        children = list(model.named_children())
        if children:
            return children[-1][0]
        return None


def save_full_model(
    model,
    model_type,
    arch_type,
    num_patches,
    output,
    model_filename,
    actual_input_size,
    gradcam_layer,
    test_metrics,
    device,
):
    """Save full model with metadata for patch-based models"""
    if isinstance(actual_input_size, int):
        actual_input_size = (actual_input_size, actual_input_size)

    # Create dummy input with patch dimensions
    dummy_input = torch.randn(
        1, num_patches, 3, actual_input_size[0], actual_input_size[1]
    ).to(device)

    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    model_to_save.eval()
    with torch.no_grad():
        start = time.time()
        _ = model_to_save(dummy_input)
        inference_time = time.time() - start

    model_name = type(model_to_save).__name__
    num_params = sum(p.numel() for p in model_to_save.parameters())

    full_model_dir = os.path.join(output, "models")
    os.makedirs(full_model_dir, exist_ok=True)
    full_model_path = os.path.join(full_model_dir, f"{model_filename}_full.pth")

    model_metadata = {
        "model": model_to_save,
        "model_type": model_type,
        "model_name": model_name,
        "arch_type": arch_type,
        "num_params": num_params,
        "input_size": actual_input_size,
        "num_patches": num_patches,
        "gradcam_layer": gradcam_layer,
        "test_metrics": test_metrics,
        "inference_time": inference_time,
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }

    try:
        torch.save(model_metadata, full_model_path)
        print(f"\n✅ Saved full model to: {full_model_path}")
        print(f"   Model name: {model_metadata['model_name']}")
        print(f"   Model type: {model_metadata['model_type']}")
        print(f"   Architecture: {model_metadata['arch_type']}")
        print(f"   Num params: {model_metadata['num_params']}")
        print(f"   Input size: {model_metadata['input_size']}")
        print(f"   Num patches: {model_metadata['num_patches']}")
        print(f"   GradCAM layer: {model_metadata['gradcam_layer']}")
        print(f"   Inference time: {model_metadata['inference_time']:.4f}s")
        print(f"   Test Accuracy: {model_metadata['test_metrics']['accuracy']:.2f}%")
        if model_metadata["test_metrics"].get("auc"):
            print(f"   Test AUC: {model_metadata['test_metrics']['auc']:.4f}")
        print(
            f"   Test Precision: {model_metadata['test_metrics'].get('precision', 0.0):.2f}%"
        )
        print(
            f"   Test Recall: {model_metadata['test_metrics'].get('recall', 0.0):.2f}%"
        )
    except Exception as e:
        print(f"⚠️ Error saving full model: {e}")


def generate_visualizations(
    model,
    test_loader,
    test_df,
    image_info,
    class_names,
    output,
    model_filename,
    batch_size,
    actual_input_size,
    num_patches,
    use_otsu,
    use_gradcam,
    gradcam_layer,
    device,
):
    """Generate visualizations for test set (patch-based models)"""
    vis_dir = os.path.join(output, "test_patch", model_filename)
    os.makedirs(vis_dir, exist_ok=True)

    test_image_ids = test_df["image_id"].tolist()
    model.eval()

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + images.shape[0], len(test_image_ids))
        batch_image_ids = test_image_ids[start_idx:end_idx]

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        for i in range(images.shape[0]):
            if i >= len(batch_image_ids):
                continue

            image_id = str(batch_image_ids[i])
            pred_class = predicted[i].item()
            gt_label = labels[i].item()
            pred_prob = probs[i, pred_class].item()

            if image_id not in image_info:
                continue

            info = image_info[image_id]
            original_size = info["original_size"]
            image_path = info["image_path"]
            gt_bboxes = info.get("bbx_list", None)

            # Generate GradCAM for MIL model (returns array of heatmaps, one per patch)
            gradcam_maps = None
            if use_gradcam and gradcam_layer:
                try:
                    # Use entire patch tensor for MIL GradCAM
                    input_tensor = images[i : i + 1].clone().requires_grad_(True)
                    test_model = (
                        model.module if isinstance(model, nn.DataParallel) else model
                    )
                    with torch.set_grad_enabled(True):
                        result = mil_gradcam(
                            test_model,
                            input_tensor,
                            gradcam_layer,
                            class_idx=pred_class,
                        )
                        # result is ndarray with shape [num_patches, H, W]
                        if isinstance(result, np.ndarray) and result.ndim == 3:
                            gradcam_maps = result
                        elif isinstance(result, np.ndarray) and result.ndim == 2:
                            # Single heatmap fallback
                            gradcam_maps = np.expand_dims(result, axis=0)
                except Exception as e:
                    print(f"⚠️ MIL GradCAM failed for {image_id}: {e}")
                    import traceback

                    traceback.print_exc()

            # Merge patch heatmaps if available
            combined_gradcam = None
            if gradcam_maps is not None and gradcam_maps.ndim == 3:
                try:
                    # Load original image to get dimensions
                    from PIL import Image

                    orig_img = Image.open(image_path).convert("RGB")
                    img_np = np.array(orig_img)
                    height, width = img_np.shape[:2]

                    # Calculate patch positions (same logic as split_image_into_patches)
                    overlap_ratio = 0.2
                    num_patches_actual = gradcam_maps.shape[0]
                    patch_height = height // num_patches
                    step = int(patch_height * (1 - overlap_ratio))

                    if num_patches == 1 or step <= 0:
                        starts = [0]
                    else:
                        starts = [i * step for i in range(num_patches - 1)]
                        starts.append(height - patch_height)

                    # Merge heatmaps vertically with max overlap
                    combined_gradcam = np.zeros((height, width), dtype=np.uint8)
                    for idx in range(min(num_patches_actual, len(starts))):
                        cam = gradcam_maps[idx]
                        start_h = starts[idx]
                        end_h = start_h + patch_height
                        if idx == num_patches - 1:
                            start_h = height - patch_height
                            end_h = height

                        patch_h = end_h - start_h
                        cam_img = Image.fromarray(cam).resize(
                            (width, patch_h), Image.Resampling.BILINEAR
                        )
                        cam_np = np.array(cam_img)
                        combined_gradcam[start_h:end_h, :] = np.maximum(
                            combined_gradcam[start_h:end_h, :], cam_np
                        )

                    # Resize to original size
                    combined_gradcam_img = Image.fromarray(combined_gradcam).resize(
                        (width, height), Image.Resampling.BILINEAR
                    )
                    combined_gradcam = np.array(combined_gradcam_img)

                except Exception as e:
                    print(f"⚠️ Error merging GradCAM maps for {image_id}: {e}")
                    # Fallback: use first patch
                    combined_gradcam = (
                        gradcam_maps[0] if gradcam_maps.shape[0] > 0 else None
                    )

            # Visualize using combined heatmap
            save_path = os.path.join(vis_dir, f"{image_id}.png")
            try:
                # Get first patch for visualization (as tensor reference)
                first_patch = images[i, 0] if images.dim() == 5 else images[i]
                visualize_classification_result(
                    first_patch,
                    gt_bboxes,
                    pred_class,
                    gt_label,
                    pred_prob,
                    combined_gradcam,  # Use combined heatmap instead of single patch
                    save_path,
                    class_names,
                    original_size,
                    image_path=image_path,
                    use_otsu=use_otsu,
                )
            except Exception as e:
                print(f"⚠️ Error visualizing {image_id}: {e}")
                import traceback

                traceback.print_exc()

    print(f"✅ Saved visualizations to: {vis_dir}")


def run_patch_test(
    data_folder,
    model_type,
    arch_type,
    num_patches,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,
    save_visualizations=True,
    only_viz=False,
    use_gradcam=True,
    use_otsu=True,
):
    """Run patch-based/MIL classification model testing with GradCAM"""
    config = load_config(config_path)

    train_df, test_df, class_names = load_metadata(
        data_folder, config_path, target_column=target_column
    )

    if class_names and isinstance(class_names[0], int):
        class_names = [str(c) for c in class_names]

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Test samples: {len(test_df)}")
    print(f"Architecture: {arch_type}")
    print(f"Number of patches: {num_patches}")

    # Load image metadata
    image_info = load_image_metadata(data_folder, test_df)

    _, test_loader = get_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        num_patches=num_patches,
        img_size=img_size,
    )

    model = get_patch_model(
        model_type=model_type,
        num_patches=num_patches,
        arch_type=arch_type,
        num_classes=len(class_names),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not pretrained_model_path:
        print("⚠️ No pretrained model path provided!")
        return

    try:
        model.load_state_dict(
            torch.load(pretrained_model_path, map_location=device),
        )
        print(f"✅ Loaded pretrained model from {pretrained_model_path}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    try:
        sample_batch = next(iter(test_loader))
        images, _ = sample_batch
        # For patch models: [B, num_patches, C, H, W]
        actual_input_size = (images.shape[3], images.shape[4])
    except:
        actual_input_size = img_size if img_size else (448, 448)

    model_filename = os.path.basename(pretrained_model_path).replace(".pth", "")

    # GradCAM setup
    gradcam_layer = None
    if use_gradcam:
        test_model = model.module if isinstance(model, nn.DataParallel) else model
        gradcam_layer = get_gradcam_layer_patch(
            test_model, model_type.lower(), arch_type
        )
        if gradcam_layer:
            print(f"✓ GradCAM enabled: {gradcam_layer}")
        else:
            use_gradcam = False
            print("⚠️ GradCAM layer not found")

    # Evaluation
    if not only_viz:
        print("\n" + "=" * 60)
        print("Evaluation on Test Set")
        print("=" * 60)

        results = evaluate_classification_model(model, test_loader, device)

        metrics = compute_classification_metrics(
            results["preds"], results["labels"], results["probs"], class_names
        )
        metrics["accuracy"] = results["accuracy"]

        print_test_metrics(metrics, class_names, results["labels"], results["preds"])

        save_full_model(
            model,
            model_type,
            arch_type,
            num_patches,
            output,
            model_filename,
            actual_input_size,
            gradcam_layer,
            metrics,
            device,
        )

    # Visualization
    if save_visualizations:
        print("\n" + "=" * 60)
        print("Generate Visualizations")
        print("=" * 60)

        generate_visualizations(
            model,
            test_loader,
            test_df,
            image_info,
            class_names,
            output,
            model_filename,
            batch_size,
            actual_input_size,
            num_patches,
            use_otsu,
            use_gradcam,
            gradcam_layer,
            device,
        )

    if not only_viz:
        return metrics["accuracy"] / 100
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument(
        "--arch_type",
        type=str,
        default="mil",
        choices=[
            "patch_resnet",
            "patch_transformer",
            "token_mixer",
            "global_local",
            "global_local_token",
            "mil",
            "mil_v2",
            "mil_v3",
            "mil_v4",
            "mil_v5",
            "mil_v6",
            "mil_v7",
            "mil_v8",
            "mil_v9",
            "mil_v10",
            "mil_v11",
            "mil_v12",
        ],
    )
    parser.add_argument("--num_patches", type=int, default=2)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--img_size", type=str)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--only_viz", action="store_true")
    parser.add_argument("--no_gradcam", action="store_true")
    parser.add_argument("--no_otsu", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    run_patch_test(
        data_folder=get_arg_or_config(
            args.data_folder, config.get("data_folder"), None
        ),
        model_type=get_arg_or_config(args.model_type, config.get("model_type"), None),
        arch_type=get_arg_or_config(args.arch_type, config.get("arch_type"), "mil"),
        num_patches=get_arg_or_config(args.num_patches, config.get("num_patches"), 2),
        batch_size=get_arg_or_config(args.batch_size, config.get("batch_size"), 16),
        output=get_arg_or_config(args.output, config.get("output"), "output"),
        config_path=args.config,
        img_size=parse_img_size(
            get_arg_or_config(args.img_size, config.get("image_size"), None)
        ),
        pretrained_model_path=args.pretrained_model_path,
        target_column=get_arg_or_config(
            args.target_column, config.get("target_column"), None
        ),
        save_visualizations=not args.no_viz,
        only_viz=args.only_viz,
        use_gradcam=not args.no_gradcam,
        use_otsu=not args.no_otsu,
    )
