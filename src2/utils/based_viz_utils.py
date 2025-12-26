"""
Visualization utilities for classification models
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pandas as pd


def denormalize_image(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize image tensor to [0, 1]"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def load_original_image(image_path, original_size, fallback_tensor=None):
    """Load and resize original image"""
    orig_h, orig_w = original_size

    if image_path and os.path.exists(image_path):
        img_original = Image.open(image_path).convert("RGB")
        if img_original.size != (orig_w, orig_h):
            img_original = img_original.resize(
                (orig_w, orig_h), Image.Resampling.BILINEAR
            )
        return np.array(img_original).astype(np.float32) / 255.0

    if fallback_tensor is not None:
        img_denorm = denormalize_image(
            fallback_tensor.cpu(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_original = img_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        return np.array(img_original).astype(np.float32) / 255.0

    return None


def draw_bboxes_on_axis(ax, bbox_list, original_size, color="lime", linewidth=2):
    """Draw bounding boxes on axis"""
    orig_h, orig_w = original_size
    num_valid = 0

    if bbox_list is not None and len(bbox_list) > 0:
        for bbox_data in bbox_list:
            if (
                isinstance(bbox_data, (list, np.ndarray))
                and len(bbox_data) == 4
                and not any(pd.isna(v) for v in bbox_data)
            ):
                x, y, w, h = bbox_data
                if (
                    x >= 0
                    and y >= 0
                    and x + w <= orig_w
                    and y + h <= orig_h
                    and w > 0
                    and h > 0
                ):
                    rect = mpatches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=linewidth,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    num_valid += 1

    return num_valid


def apply_otsu_threshold(heatmap):
    """Apply Otsu thresholding to heatmap"""
    import cv2

    _, binary = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def visualize_classification_result(
    image_tensor,
    gt_bboxes,
    pred_class,
    gt_label,
    pred_prob,
    gradcam_map,
    save_path,
    class_names,
    original_size,
    image_path=None,
    use_otsu=False,
):
    """Create classification visualization with 3 panels"""
    import cv2

    img_original_np = load_original_image(image_path, original_size, image_tensor)
    if img_original_np is None:
        return

    # Prepare GradCAM overlay
    gradcam_blend = None
    gradcam_otsu_overlay = None
    if gradcam_map is not None:
        # Resize GradCAM to original size
        gradcam_resized = cv2.resize(gradcam_map, original_size[::-1])

        # Create colored heatmap
        gradcam_colored = cv2.applyColorMap(gradcam_resized, cv2.COLORMAP_JET)
        gradcam_colored = cv2.cvtColor(gradcam_colored, cv2.COLOR_BGR2RGB) / 255.0
        gradcam_blend = np.clip(0.6 * img_original_np + 0.4 * gradcam_colored, 0, 1)

        # Otsu threshold
        if use_otsu:
            gradcam_otsu = apply_otsu_threshold(gradcam_resized)
            gradcam_otsu_overlay = img_original_np.copy()

    # Create figure with 3 panels
    num_plots = 3 if use_otsu and gradcam_otsu_overlay is not None else 2
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))

    if num_plots == 2:
        ax1, ax2 = axs
    else:
        ax1, ax2, ax3 = axs

    # Panel 1: Original with GT bboxes
    ax1.imshow(img_original_np)
    num_valid = draw_bboxes_on_axis(ax1, gt_bboxes, original_size, "lime", 2)
    title_left = f"GT: {class_names[gt_label]}"
    title_left += f" | {num_valid} bbox(es)" if num_valid > 0 else " (No bbox)"
    ax1.set_title(title_left, fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2: GradCAM overlay
    if gradcam_blend is not None:
        ax2.imshow(gradcam_blend)
    else:
        ax2.imshow(img_original_np)

    pred_label = class_names[pred_class]
    color = "green" if pred_class == gt_label else "red"
    title_right = f"Pred: {pred_label} | {pred_prob:.3f}"
    ax2.set_title(title_right, fontsize=12, fontweight="bold", color=color)
    ax2.axis("off")

    # Panel 3: GradCAM + Otsu (optional)
    if num_plots == 3 and gradcam_otsu_overlay is not None:
        ax3.imshow(gradcam_otsu_overlay)
        ax3.imshow(gradcam_otsu / 255.0, cmap="gray", alpha=0.5)
        ax3.set_title("GradCAM + Otsu", fontsize=12, fontweight="bold")
        ax3.axis("off")

    plt.tight_layout()
    save_path = clean_image_filename(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def clean_image_filename(filename):
    """Remove duplicate .png extension"""
    if filename.endswith(".png.png"):
        return filename[: filename.find(".png") + 4]
    return filename
