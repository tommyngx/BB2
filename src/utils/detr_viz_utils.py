"""
Visualization utilities for DETR models
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pandas as pd

from src.utils.detr_common_utils import bbox_to_pixel, compute_bbox_confidence
from src.utils.detr_attn_map import prepare_attention_heatmap, create_heatmap_overlay


def denormalize_image(tensor, mean, std):
    """Denormalize image tensor"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    return torch.clamp(img, 0, 1)


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
    """Draw bounding boxes"""
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


def draw_predicted_bboxes(
    ax, pred_bboxes, pred_scores, original_size, gt_bbox_list=None, obj_threshold=0.5
):
    """Draw predicted bboxes"""
    orig_h, orig_w = original_size
    num_pred_boxes = 0
    max_iou = 0.0

    if pred_bboxes is None or pred_scores is None:
        return num_pred_boxes, max_iou

    valid_mask = pred_scores >= obj_threshold

    for idx in range(len(pred_bboxes)):
        if not valid_mask[idx]:
            continue

        bbox = pred_bboxes[idx]
        obj_score = pred_scores[idx].item()

        if not torch.isnan(bbox).any():
            x_pix, y_pix, w_pix, h_pix = bbox_to_pixel(
                bbox.cpu().numpy(), original_size
            )

            if (
                x_pix >= 0
                and y_pix >= 0
                and x_pix + w_pix <= orig_w
                and y_pix + h_pix <= orig_h
                and w_pix > 0
                and h_pix > 0
            ):
                color = (
                    "red"
                    if obj_score >= 0.8
                    else "orange"
                    if obj_score >= 0.6
                    else "yellow"
                )

                rect = mpatches.Rectangle(
                    (x_pix, y_pix),
                    w_pix,
                    h_pix,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                ax.text(
                    x_pix,
                    y_pix - 5,
                    f"{obj_score:.2f}",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )
                num_pred_boxes += 1

                if gt_bbox_list is not None and len(gt_bbox_list) > 0:
                    try:
                        gt_first = gt_bbox_list[0]
                        gt_norm = torch.tensor(
                            [
                                gt_first[0] / orig_w,
                                gt_first[1] / orig_h,
                                gt_first[2] / orig_w,
                                gt_first[3] / orig_h,
                            ],
                            device=bbox.device,
                        )
                        iou = compute_bbox_confidence(bbox, gt_norm)
                        max_iou = max(max_iou, iou)
                    except:
                        pass

    return num_pred_boxes, max_iou


def visualize_detr_result(
    image_tensor,
    attn_map,
    pred_bboxes,
    pred_obj_scores,
    gt_bbox_list,
    pred_class,
    gt_label,
    pred_prob,
    save_path,
    class_names,
    original_size,
    image_path=None,
    obj_threshold=0.5,
    use_otsu=False,
    gradcam_map=None,
):
    """Create DETR visualization"""

    img_original_np = load_original_image(image_path, original_size, image_tensor)
    if img_original_np is None:
        return

    attn_resized_np, mask = prepare_attention_heatmap(attn_map, original_size, use_otsu)
    blend_img = create_heatmap_overlay(img_original_np, attn_resized_np, mask)

    gradcam_blend = None
    if gradcam_map is not None:
        gradcam_resized = Image.fromarray(gradcam_map).resize(
            original_size[::-1], Image.Resampling.BILINEAR
        )
        gradcam_np = np.array(gradcam_resized)
        gradcam_color = plt.cm.jet(gradcam_np / 255.0)[..., :3]
        gradcam_blend = np.clip(0.6 * img_original_np + 0.4 * gradcam_color, 0, 1)

    num_plots = 3 if gradcam_map is not None else 2
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))

    if num_plots == 2:
        ax1, ax2 = axs
    else:
        ax1, ax2, ax3 = axs

    # Panel 1
    ax1.imshow(img_original_np)
    num_valid = draw_bboxes_on_axis(ax1, gt_bbox_list, original_size, "lime")
    title_left = f"GT: {class_names[gt_label]}"
    title_left += f" | {num_valid} bbox(es)" if num_valid > 0 else " (No bbox)"
    ax1.set_title(title_left, fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2
    ax2.imshow(blend_img)
    num_pred, max_iou = draw_predicted_bboxes(
        ax2, pred_bboxes, pred_obj_scores, original_size, gt_bbox_list, obj_threshold
    )
    title_right = (
        f"DETR: {class_names[pred_class]} | {pred_prob:.3f} | Boxes: {num_pred}"
    )
    if max_iou > 0:
        title_right += f" | IoU: {max_iou:.3f}"
    ax2.set_title(title_right, fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Panel 3
    if gradcam_blend is not None:
        ax3.imshow(gradcam_blend)
        ax3.set_title(
            f"GradCAM: {class_names[pred_class]}", fontsize=12, fontweight="bold"
        )
        ax3.axis("off")

    plt.tight_layout()
    save_path = clean_image_filename(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def clean_image_filename(filename):
    """
    Nếu tên file có đuôi .png.png thì chỉ giữ lại phần trước .png đầu tiên.
    """
    if filename.endswith(".png.png"):
        return filename[: filename.find(".png") + 4]
    return filename
