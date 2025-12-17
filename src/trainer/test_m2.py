import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
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
from src.data.dataloader import load_metadata
from src.models.m2_model import get_m2_model
from src.trainer.m2_evaluate import evaluate_m2_model
from src.utils.common import load_config, get_arg_or_config
from src.trainer.train_based import get_gradcam_layer


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
    img = torch.clamp(img, 0, 1)  # Ensure values are in [0, 1]
    return img


def bbox_to_xyxy(bbox, original_size):
    """
    Convert normalized bbox [x, y, w, h] to pixel coordinates [x1, y1, x2, y2]
    in original image size

    Args:
        bbox: normalized bbox [x, y, w, h] in range [0, 1]
        original_size: (height, width) of original image
    """
    x, y, w, h = bbox
    orig_h, orig_w = original_size

    # Convert from normalized coordinates to original image pixel coordinates
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
    original_size,  # (height, width) of original image
):
    """
    Create side-by-side visualization:
    Left: Original image + GT bbox
    Right: Attention heatmap with Otsu + Predicted bbox

    Args:
        original_size: (height, width) of the original image before resizing
    """
    # Denormalize image using m2_data normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_denorm = denormalize_image(image_tensor.cpu(), mean, std)

    # Convert to numpy
    img_np = img_denorm.permute(1, 2, 0).numpy()

    # Resize image to original size for visualization
    orig_h, orig_w = original_size
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_original_size = img_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    img_original_np = np.array(img_original_size).astype(np.float32) / 255.0

    # Process attention map - resize to original size
    attn_np = attn_map.squeeze().cpu().numpy()
    attn_resized = Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    attn_resized_np = np.array(attn_resized)

    # Apply Otsu threshold
    otsu_thresh = threshold_otsu(attn_resized_np)
    mask = attn_resized_np > otsu_thresh

    # Create heatmap overlay on original size image
    cam_color = plt.cm.jet(attn_resized_np / 255.0)[..., :3]
    blend_img = img_original_np.copy()
    blend_alpha = 0.6
    blend_img[mask] = (1 - blend_alpha) * blend_img[mask] + blend_alpha * cam_color[
        mask
    ]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original image + GT bbox (in original size)
    ax1.imshow(img_original_np)
    if gt_bbox is not None and not torch.isnan(gt_bbox).any():
        # Convert bbox to original image coordinates
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

    # Right: Heatmap + Predicted bbox (in original size)
    ax2.imshow(blend_img)
    if pred_bbox is not None:
        # Convert bbox to original image coordinates
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
    """
    Test M2Model with visualization of results
    """
    # Load config and data
    config = load_config(config_path)
    train_df, test_df, class_names = load_metadata(
        data_folder, config_path, target_column=target_column
    )

    _, test_loader = get_m2_dataloaders(
        train_df,
        test_df,
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

    # Task 1: Evaluate on test set
    print("\n" + "=" * 50)
    print("Task 1: Evaluation on Test Set")
    print("=" * 50)
    test_loss, test_acc, test_iou = evaluate_m2_model(
        model,
        test_loader,
        device=device,
        mode="Test",
        return_loss=True,
        lambda_bbox=lambda_bbox,
    )

    # Task 2: Save full model
    print("\n" + "=" * 50)
    print("Task 2: Save Full Model")
    print("=" * 50)

    # Extract model name from pretrained path
    model_filename = os.path.basename(pretrained_model_path).replace(".pth", "")

    # Get normalize params - use m2_data default
    normalize_params = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

    # Use get_gradcam_layer function to determine the correct layer
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

    # Save state_dict + metadata instead of full model to avoid pickling issues
    checkpoint = {
        "state_dict": model_to_save.state_dict(),
        "model_type": model_type,
        "num_classes": len(class_names),
        "input_size": actual_input_size,
        "model_name": model_type,
        "gradcam_layer": gradcam_layer,
        "normalize": normalize_params,
        "num_patches": None,
        "arch_type": "m2",
        "class_names": class_names,
    }

    full_model_dir = os.path.join(output, "models")
    os.makedirs(full_model_dir, exist_ok=True)
    full_model_path = os.path.join(full_model_dir, f"{model_filename}_full.pth")

    try:
        torch.save(checkpoint, full_model_path)
        print(f"✅ Saved full model to: {full_model_path}")
        print(f"   GradCAM layer: {gradcam_layer}")
        print(f"   Model type: {model_type}")
        print(f"   Input size: {actual_input_size}")
    except Exception as e:
        print(f"⚠️ Error saving full model: {e}")

    # Task 3: Visualize results
    if save_visualizations:
        print("\n" + "=" * 50)
        print("Task 3: Generate Visualizations")
        print("=" * 50)

        # Create output directory
        vis_dir = os.path.join(output, "test", model_filename)
        os.makedirs(vis_dir, exist_ok=True)

        # Load original image sizes from metadata
        metadata_path = os.path.join(data_folder, "metadata.csv")
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(data_folder, "metadata2.csv")

        original_sizes = {}
        if os.path.exists(metadata_path):
            meta_df = pd.read_csv(metadata_path)
            for _, row in meta_df.iterrows():
                img_id = row.get("image_id", None)
                if img_id:
                    # Try to get original dimensions
                    orig_h = row.get("original_height", None)
                    orig_w = row.get("original_width", None)
                    if orig_h and orig_w:
                        original_sizes[img_id] = (int(orig_h), int(orig_w))

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                has_bbox = batch["has_bbox"].to(device)
                image_ids = batch.get(
                    "image_id", [f"img_{batch_idx}_{i}" for i in range(len(images))]
                )

                # Forward pass
                outputs = model(images)
                if len(outputs) == 3:
                    cls_outputs, bbox_outputs, attn_maps = outputs
                else:
                    cls_outputs, bbox_outputs = outputs
                    attn_maps = None

                # Get predictions
                _, predicted = torch.max(cls_outputs, 1)
                probs = torch.softmax(cls_outputs, dim=1)

                # Process each image in batch
                for i in range(len(images)):
                    image_id = (
                        image_ids[i]
                        if isinstance(image_ids, list)
                        else str(image_ids[i].item())
                    )
                    pred_class = predicted[i].item()
                    gt_label = labels[i].item()
                    pred_prob = probs[i, pred_class].item()

                    # Get bbox
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

                    # Skip if no attention map
                    if attn_map is None:
                        continue

                    # Get original image size
                    if image_id in original_sizes:
                        original_size = original_sizes[image_id]
                    else:
                        # Fallback: try to load original image to get size
                        test_row = test_df[test_df["image_id"] == image_id]
                        if not test_row.empty:
                            img_path = os.path.join(
                                data_folder, test_row.iloc[0]["link"]
                            )
                            if os.path.exists(img_path):
                                with Image.open(img_path) as img_orig:
                                    original_size = (img_orig.height, img_orig.width)
                            else:
                                original_size = actual_input_size
                        else:
                            original_size = actual_input_size

                    # Save visualization
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
                    )

        print(f"✅ Saved visualizations to: {vis_dir}")

    return test_loss, test_acc, test_iou


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
