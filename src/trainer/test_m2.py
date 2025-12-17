import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd

from src.data.m2_data import get_m2_dataloaders
from src.models.m2_model import get_m2_model
from src.utils.common import load_config, get_arg_or_config
from src.trainer.train_based import get_gradcam_layer


def load_data_bbx3(data_folder):
    """Load metadata with bounding box information grouped by image_id"""
    metadata_path = os.path.join(data_folder, "metadata2.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(data_folder, "metadata.csv")

    if not os.path.exists(metadata_path):
        return None, None, {}

    df = pd.read_csv(metadata_path)

    # Create bbx column from x, y, width, height columns
    if all(col in df.columns for col in ["x", "y", "width", "height"]):
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

    # Create mapping: image_id -> (original_size, image_path, bbx_list)
    image_info = {}
    for image_id in df["image_id"].unique():
        img_rows = df[df["image_id"] == image_id]
        first_row = img_rows.iloc[0]

        # Get image path
        img_path = os.path.join(data_folder, first_row["link"])

        # Get original size
        if "original_height" in first_row and "original_width" in first_row:
            orig_h = int(first_row["original_height"])
            orig_w = int(first_row["original_width"])
            original_size = (orig_h, orig_w)
        else:
            # Load image to get size
            if os.path.exists(img_path):
                with Image.open(img_path) as img_orig:
                    original_size = (img_orig.height, img_orig.width)
            else:
                original_size = None

        # Get bbox list
        bbx_list = first_row.get("bbx_list", None)

        image_info[image_id] = {
            "original_size": original_size,
            "image_path": img_path,
            "bbx_list": bbx_list,
        }

    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    test_df = df[df["split"] == "test"] if "split" in df.columns else df

    return train_df, test_df, image_info


def parse_img_size(val):
    if val is None:
        return None
    if "x" in val:
        h, w = val.lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def denormalize_image(tensor, mean, std):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def bbox_to_xyxy(bbox, original_size):
    """Convert normalized bbox [x, y, w, h] to pixel coordinates [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    orig_h, orig_w = original_size
    x1 = int(x * orig_w)
    y1 = int(y * orig_h)
    x2 = int((x + w) * orig_w)
    y2 = int((y + h) * orig_h)
    return x1, y1, x2, y2


def compute_bbox_confidence(pred_bbox, gt_bbox):
    """Compute IoU as bbox confidence"""
    from src.utils.m2_utils import compute_iou

    iou = compute_iou(pred_bbox.unsqueeze(0), gt_bbox.unsqueeze(0))
    return iou.item()


def visualize_m2_result(
    image_tensor,
    attn_map,
    pred_bbox,
    gt_bbox,
    pred_class,
    gt_label,
    pred_prob,
    bbox_conf,
    save_path,
    class_names,
    original_size,
    image_path=None,
):
    """Create side-by-side visualization with original image size"""
    orig_h, orig_w = original_size

    # ALWAYS load original image from disk for accurate visualization
    if image_path and os.path.exists(image_path):
        img_original = Image.open(image_path).convert("RGB")
        # Ensure size matches metadata
        if img_original.size != (orig_w, orig_h):
            img_original = img_original.resize(
                (orig_w, orig_h), Image.Resampling.BILINEAR
            )
        img_original_np = np.array(img_original).astype(np.float32) / 255.0
    else:
        # Fallback: denormalize tensor (less accurate)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        img_denorm = denormalize_image(image_tensor.cpu(), mean, std)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_original = img_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        img_original_np = np.array(img_original).astype(np.float32) / 255.0

    # Resize attention map to original size
    attn_np = attn_map.squeeze().cpu().numpy()
    attn_resized = Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    attn_resized_np = np.array(attn_resized)

    # Apply Otsu threshold
    otsu_thresh = threshold_otsu(attn_resized_np)
    mask = attn_resized_np > otsu_thresh

    # Create heatmap overlay
    cam_color = plt.cm.jet(attn_resized_np / 255.0)[..., :3]
    blend_img = img_original_np.copy()
    blend_alpha = 0.6
    blend_img[mask] = (1 - blend_alpha) * blend_img[mask] + blend_alpha * cam_color[
        mask
    ]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original image + GT bbox
    ax1.imshow(img_original_np)
    if gt_bbox is not None and not torch.isnan(gt_bbox).any():
        # Scale bbox to original image size
        x1, y1, x2, y2 = bbox_to_xyxy(gt_bbox.cpu().numpy(), original_size)
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax1.add_patch(rect)
        title_left = f"Original ({orig_w}x{orig_h}) | GT: {class_names[gt_label]}"
    else:
        title_left = (
            f"Original ({orig_w}x{orig_h}) | GT: {class_names[gt_label]} (No bbox)"
        )
    ax1.set_title(title_left, fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Right: Heatmap + Predicted bbox
    ax2.imshow(blend_img)
    if pred_bbox is not None:
        # Scale bbox to original image size
        x1, y1, x2, y2 = bbox_to_xyxy(pred_bbox.cpu().numpy(), original_size)
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax2.add_patch(rect)

    pred_label_str = class_names[pred_class]
    title_right = f"Pred: {pred_label_str} | Prob: {pred_prob:.3f}"
    if bbox_conf is not None:
        title_right += f" | BBox Conf: {bbox_conf:.3f}"
    ax2.set_title(title_right, fontsize=12, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_m2_test_with_visualization(
    data_folder,
    model_type,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,
    lambda_bbox=1.0,
    save_visualizations=True,
):
    """Test M2Model with visualization of results"""
    # Load config
    config = load_config(config_path)

    # Load metadata with bbox info ONLY from load_data_bbx3
    train_df_bbx, test_df_bbx, image_info = load_data_bbx3(data_folder)

    if train_df_bbx is None or test_df_bbx is None:
        print("❌ Error: Could not load metadata from data folder")
        return

    # Get class names from train_df
    if target_column and target_column in train_df_bbx.columns:
        class_names = sorted(train_df_bbx[target_column].unique())
    elif "cancer" in train_df_bbx.columns:
        class_names = sorted(train_df_bbx["cancer"].unique())
    else:
        print("❌ Error: Could not find target column in metadata")
        return

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Train samples: {len(train_df_bbx)}, Test samples: {len(test_df_bbx)}")

    _, test_loader = get_m2_dataloaders(
        train_df_bbx,
        test_df_bbx,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode="test",
    )

    # Load model
    model = get_m2_model(model_type=model_type, num_classes=len(class_names))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_model_path:
        try:
            state_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"⚠️ Error loading pretrained model: {e}")
            return
    else:
        print("⚠️ No pretrained model path provided!")
        return

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Task 1: Evaluate on test set with tqdm
    print("\n" + "=" * 50)
    print("Task 1: Evaluation on Test Set")
    print("=" * 50)

    # Run evaluation with tqdm progress bar
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    total_iou = 0.0
    num_bbox_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            outputs = model(images)
            if len(outputs) == 3:
                cls_outputs, bbox_outputs, _ = outputs
            else:
                cls_outputs, bbox_outputs = outputs

            # Classification
            _, predicted = torch.max(cls_outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # IoU for bbox
            if has_bbox.sum() > 0:
                pos_mask = has_bbox.bool()
                from src.utils.m2_utils import compute_iou

                iou = compute_iou(bbox_outputs[pos_mask], bboxes[pos_mask])
                total_iou += iou.sum().item()
                num_bbox_samples += has_bbox.sum().item()

    test_acc = correct / total
    test_iou = total_iou / max(num_bbox_samples, 1)
    print(f"Test Accuracy: {test_acc * 100:.2f}% | IoU: {test_iou:.4f}")

    # Task 2: Save full model (like run_gradcam in train_based)
    print("\n" + "=" * 50)
    print("Task 2: Save Full Model")
    print("=" * 50)

    # Extract model name from pretrained path
    model_filename = os.path.basename(pretrained_model_path).replace(".pth", "")

    # Get normalize params - use m2_data default
    normalize_params = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

    # Use get_gradcam_layer function
    model_name = model_type.lower()
    if isinstance(model, nn.DataParallel):
        gradcam_layer = get_gradcam_layer(model.module, model_name)
        model_to_save = model.module
    else:
        gradcam_layer = get_gradcam_layer(model, model_name)
        model_to_save = model

    # Get actual input size from test_loader
    try:
        sample_batch = next(iter(test_loader))
        actual_input_size = (
            sample_batch["image"].shape[2],
            sample_batch["image"].shape[3],
        )
    except Exception:
        actual_input_size = img_size if img_size else (448, 448)

    # Calculate real inference time
    if isinstance(actual_input_size, int):
        actual_input_size = (actual_input_size, actual_input_size)
    dummy_input = torch.randn(1, 3, actual_input_size[0], actual_input_size[1]).to(
        device
    )
    model_to_save.eval()
    with torch.no_grad():
        start = time.time()
        _ = model_to_save(dummy_input)
        end = time.time()
        inference_time = end - start

    # Save only state_dict and all metadata (avoid pickling local classes)
    model_info = {
        "state_dict": model_to_save.state_dict(),
        "input_size": actual_input_size,
        "gradcam_layer": gradcam_layer,
        "model_name": model_type,
        "normalize": normalize_params,
        "inference_time": inference_time,
        "num_patches": None,
        "arch_type": "m2",
        "class_names": class_names,
    }

    full_model_dir = os.path.join(output, "models")
    os.makedirs(full_model_dir, exist_ok=True)
    full_model_path = os.path.join(full_model_dir, f"{model_filename}_full.pth")

    try:
        torch.save(model_info, full_model_path)
        print(f"✅ Saved full model info (state_dict + meta) to: {full_model_path}")
        print(f"   Model name: {model_type}")
        print(f"   Input size: {actual_input_size}")
        print(f"   GradCAM layer: {gradcam_layer}")
        print(f"   Inference time: {inference_time:.4f}s")
    except Exception as e:
        print(f"⚠️ Error saving full model: {e}")

    # Task 3: Visualize results
    if save_visualizations:
        print("\n" + "=" * 50)
        print("Task 3: Generate Visualizations")
        print("=" * 50)

        vis_dir = os.path.join(output, "test", model_filename)
        os.makedirs(vis_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                has_bbox = batch["has_bbox"].to(device)
                image_ids = batch.get("image_id", None)

                # Debug: check if image_ids are loaded correctly
                if image_ids is None:
                    print(
                        f"⚠️ Warning: No image_id in batch {batch_idx}, skipping visualization"
                    )
                    continue

                outputs = model(images)
                if len(outputs) == 3:
                    cls_outputs, bbox_outputs, attn_maps = outputs
                else:
                    cls_outputs, bbox_outputs = outputs
                    attn_maps = None

                _, predicted = torch.max(cls_outputs, 1)
                probs = torch.softmax(cls_outputs, dim=1)

                for i in range(len(images)):
                    # Get image_id - should already be string from dataloader
                    if isinstance(image_ids, list):
                        image_id = str(image_ids[i])
                    else:
                        # Tensor case
                        image_id = (
                            str(image_ids[i].item())
                            if image_ids[i].dim() == 0
                            else str(image_ids[i])
                        )

                    pred_class = predicted[i].item()
                    gt_label = labels[i].item()
                    pred_prob = probs[i, pred_class].item()
                    pred_bbox = bbox_outputs[i] if bbox_outputs is not None else None
                    gt_bbox = bboxes[i] if has_bbox[i].item() > 0 else None

                    # Compute bbox confidence
                    bbox_conf = None
                    if (
                        pred_bbox is not None
                        and gt_bbox is not None
                        and not torch.isnan(gt_bbox).any()
                    ):
                        bbox_conf = compute_bbox_confidence(pred_bbox, gt_bbox)

                    # Get attention map
                    attn_map = attn_maps[i] if attn_maps is not None else None
                    if attn_map is None:
                        continue

                    # Get original image info from image_info dict
                    if image_id in image_info:
                        info = image_info[image_id]
                        original_size = info["original_size"]
                        image_path = info["image_path"]
                    else:
                        # Fallback: use actual_input_size
                        print(
                            f"⚠️ Warning: image_id '{image_id}' not found in image_info dict"
                        )
                        print(
                            f"   Available keys (first 5): {list(image_info.keys())[:5]}"
                        )
                        original_size = actual_input_size
                        image_path = None

                    # Skip if no original size
                    if original_size is None:
                        print(
                            f"⚠️ Warning: Could not determine original size for {image_id}"
                        )
                        continue

                    # Save visualization with image_id as filename
                    save_path = os.path.join(vis_dir, f"{image_id}.png")
                    visualize_m2_result(
                        images[i],
                        attn_map,
                        pred_bbox,
                        gt_bbox,
                        pred_class,
                        gt_label,
                        pred_prob,
                        bbox_conf,
                        save_path,
                        class_names,
                        original_size,
                        image_path=image_path,
                    )

        print(f"✅ Saved visualizations to: {vis_dir}")

    return test_acc, test_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--img_size", type=str, default=None)
    parser.add_argument("--target_column", type=str, default=None)
    parser.add_argument("--lambda_bbox", type=float, default=1.0)
    parser.add_argument("--no_viz", action="store_true", help="Skip visualizations")

    args = parser.parse_args()
    config = load_config(args.config)

    data_folder = get_arg_or_config(args.data_folder, config.get("data_folder"), None)
    model_type = get_arg_or_config(args.model_type, config.get("model_type"), None)
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    pretrained_model_path = args.pretrained_model_path
    output = get_arg_or_config(args.output, config.get("output"), "output")
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    target_column = get_arg_or_config(
        args.target_column, config.get("target_column"), None
    )
    lambda_bbox = get_arg_or_config(args.lambda_bbox, config.get("lambda_bbox"), 1.0)

    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    run_m2_test_with_visualization(
        data_folder=data_folder,
        model_type=model_type,
        batch_size=batch_size,
        output=output,
        config_path=args.config,
        img_size=img_size,
        pretrained_model_path=pretrained_model_path,
        target_column=target_column,
        lambda_bbox=lambda_bbox,
        save_visualizations=not args.no_viz,
    )
