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

from src.data.m2_data_detr import get_m2_detr_dataloaders
from src.data.dataloader import load_metadata_detr
from src.models.m2_detr_model import get_m2_detr_model
from src.utils.common import load_config, get_arg_or_config
from src.trainer.train_based import get_gradcam_layer
from src.utils.detr_utils import bbox_to_pixel, compute_bbox_confidence

# ADDED: Import for GradCAM
from src.gradcam.gradcam_utils_based import gradcam


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


def visualize_m2_detr_result(
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
    gradcam_map=None,  # ADDED: Optional GradCAM heatmap
):
    """Create visualization with original, DETR prediction, and optional GradCAM"""
    orig_h, orig_w = original_size

    # Load original image
    if image_path and os.path.exists(image_path):
        img_original = Image.open(image_path).convert("RGB")
        if img_original.size != (orig_w, orig_h):
            img_original = img_original.resize(
                (orig_w, orig_h), Image.Resampling.BILINEAR
            )
        img_original_np = np.array(img_original).astype(np.float32) / 255.0
    else:
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

    if use_otsu:
        # Apply Otsu threshold
        otsu_thresh = threshold_otsu(attn_resized_np)
        mask = attn_resized_np > otsu_thresh

        # Create heatmap overlay with masking
        cam_color = plt.cm.jet(attn_resized_np / 255.0)[..., :3]
        blend_img = img_original_np.copy()
        blend_alpha = 0.4
        blend_img[mask] = (1 - blend_alpha) * blend_img[mask] + blend_alpha * cam_color[
            mask
        ]
    else:
        # No Otsu, blend entire heatmap
        attn_normalized = attn_resized_np / 255.0
        cam_color = plt.cm.jet(attn_normalized)[..., :3]
        blend_img = img_original_np.copy()
        blend_alpha = 0.4
        blend_img = (1 - blend_alpha) * blend_img + blend_alpha * cam_color

    # ADDED: Prepare GradCAM panel if provided
    if gradcam_map is not None:
        gradcam_resized = Image.fromarray(gradcam_map).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        gradcam_np = np.array(gradcam_resized)
        gradcam_normalized = gradcam_np / 255.0
        gradcam_color = plt.cm.jet(gradcam_normalized)[..., :3]
        gradcam_blend = (
            1 - blend_alpha
        ) * img_original_np + blend_alpha * gradcam_color

    # UPDATED: Create figure with 2 or 3 subplots based on gradcam_map
    num_plots = 3 if gradcam_map is not None else 2
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))

    if num_plots == 2:
        ax1, ax2 = axs
    else:
        ax1, ax2, ax3 = axs

    # Panel 1: Original image + ALL GT bboxes
    ax1.imshow(img_original_np)
    num_valid_bboxes = 0

    if gt_bbox_list is not None and len(gt_bbox_list) > 0:
        for bbox_data in gt_bbox_list:
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
                        (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    ax1.add_patch(rect)
                    num_valid_bboxes += 1

        if num_valid_bboxes > 0:
            title_left = f"GT: {class_names[gt_label]} | {num_valid_bboxes} bbox(es)"
        else:
            title_left = f"GT: {class_names[gt_label]} (Invalid bboxes)"
    else:
        title_left = f"GT: {class_names[gt_label]} (No bbox)"

    ax1.set_title(title_left, fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2: DETR Heatmap + Predicted bboxes
    ax2.imshow(blend_img)
    num_pred_boxes = 0
    max_iou = 0.0

    if pred_bboxes is not None and pred_obj_scores is not None:
        # Filter predictions by objectness threshold
        valid_mask = pred_obj_scores >= obj_threshold

        for idx in range(len(pred_bboxes)):
            if not valid_mask[idx]:
                continue

            bbox = pred_bboxes[idx]
            obj_score = pred_obj_scores[idx].item()

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
                    # Color by confidence: red (high) to yellow (medium) to orange (low)
                    if obj_score >= 0.8:
                        color = "red"
                    elif obj_score >= 0.6:
                        color = "orange"
                    else:
                        color = "yellow"

                    rect = mpatches.Rectangle(
                        (x_pix, y_pix),
                        w_pix,
                        h_pix,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax2.add_patch(rect)

                    # Add objectness score label
                    ax2.text(
                        x_pix,
                        y_pix - 5,
                        f"{obj_score:.2f}",
                        color=color,
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                    )
                    num_pred_boxes += 1

                    # Compute IoU with GT if available
                    if gt_bbox_list is not None and len(gt_bbox_list) > 0:
                        try:
                            # Convert first GT bbox to normalized format for IoU
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
                        except Exception:
                            pass

    pred_label_str = class_names[pred_class]
    title_right = f"DETR: {pred_label_str} | {pred_prob:.3f} | Boxes: {num_pred_boxes}"
    if max_iou > 0:
        title_right += f" | IoU: {max_iou:.3f}"
    ax2.set_title(title_right, fontsize=12, fontweight="bold")
    ax2.axis("off")

    # ADDED: Panel 3: GradCAM (if available)
    if gradcam_map is not None:
        ax3.imshow(gradcam_blend)
        ax3.set_title(f"GradCAM: {pred_label_str}", fontsize=12, fontweight="bold")
        ax3.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_m2_detr_test_with_visualization(
    data_folder,
    model_type,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,
    num_queries=3,
    max_objects=3,
    save_visualizations=True,
    only_viz=False,
    sample_viz=False,
    obj_threshold=0.5,
    use_otsu=False,
    use_gradcam=False,  # ADDED: Parameter to enable GradCAM
):
    """Test M2 DETR model with visualization of multiple queries"""
    # Load config
    config = load_config(config_path)

    train_df, test_df, class_names = load_metadata_detr(
        data_folder, config_path, target_column=target_column
    )

    _, _, image_info = load_data_bbx3(data_folder)

    # FIXED: Ensure class_names is a list of strings
    if class_names and isinstance(class_names[0], int):
        class_names = [str(c) for c in class_names]

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    _, test_loader = get_m2_detr_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode="test",
        max_objects=max_objects,
    )

    # Load model
    model = get_m2_detr_model(
        model_type=model_type, num_classes=len(class_names), num_queries=num_queries
    )
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

    # Get actual input size
    try:
        sample_batch = next(iter(test_loader))
        actual_input_size = (
            sample_batch["image"].shape[2],
            sample_batch["image"].shape[3],
        )
    except Exception:
        actual_input_size = img_size if img_size else (448, 448)

    # Extract model name from pretrained path
    model_filename = os.path.basename(pretrained_model_path).replace(".pth", "")

    # Sample visualization for test data
    if sample_viz:
        from src.trainer.train_m2 import sample_viz_batches

        viz_dir = os.path.join(output, "test_sample_detr", model_filename)
        print(
            f"\n🔍 Visualizing {min(5, len(test_loader))} random test batches to {viz_dir} ..."
        )
        sample_viz_batches(test_loader, viz_dir, class_names, num_batches=5)

    # Skip Task 1 and Task 2 if only_viz is True
    if not only_viz:
        # Task 1: Evaluate on test set with FULL METRICS
        print("\n" + "=" * 50)
        print("Task 1: Evaluation on Test Set (DETR)")
        print("=" * 50)

        model.eval()
        correct, total = 0, 0
        total_iou = 0.0
        num_bbox_samples = 0

        # For detailed metrics
        all_preds = []
        all_labels = []
        all_probs = []

        # For mAP computation
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bboxes"].to(device)  # [B, M, 4]
                bbox_mask = batch["bbox_mask"].to(device)  # [B, M]

                outputs = model(images)

                # Classification
                probs = torch.softmax(outputs["cls_logits"], dim=1)
                _, predicted = torch.max(outputs["cls_logits"], 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Store for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # IoU for bbox (use top prediction by objectness)
                pred_bboxes = outputs["pred_bboxes"]  # [B, N, 4]
                pred_obj = torch.sigmoid(outputs["obj_scores"])  # [B, N, 1]

                for i in range(images.size(0)):
                    valid_mask = bbox_mask[i] > 0.5
                    if valid_mask.sum() > 0:
                        # Get top prediction
                        top_idx = pred_obj[i].squeeze(-1).argmax()
                        pred_box = pred_bboxes[i, top_idx : top_idx + 1]
                        pred_score = pred_obj[i, top_idx].item()
                        gt_boxes = bboxes[i][valid_mask]

                        from src.utils.m2_utils import compute_iou

                        iou = compute_iou(pred_box, gt_boxes[:1])
                        total_iou += iou.item()
                        num_bbox_samples += 1

                        # Store for mAP
                        all_pred_boxes.append(pred_box.cpu())
                        all_pred_scores.append(pred_score)
                        all_gt_boxes.append(gt_boxes.cpu())

        test_acc = correct / total
        test_iou = total_iou / max(num_bbox_samples, 1)

        # Compute mAP@0.5
        test_map = 0.0
        if len(all_pred_boxes) > 0:
            num_correct = 0
            for pred_box, gt_box in zip(all_pred_boxes, all_gt_boxes):
                from src.utils.m2_utils import compute_iou

                iou = compute_iou(pred_box, gt_box[:1])
                if iou.item() >= 0.5:
                    num_correct += 1
            test_map = num_correct / len(all_pred_boxes)

        # ✅ COMPUTE DETAILED METRICS (same as train)
        from sklearn.metrics import (
            classification_report,
            roc_auc_score,
            precision_recall_fscore_support,
        )
        import numpy as np

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        # AUC (only for binary classification)
        try:
            if all_probs.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1]) * 100
            else:
                auc = 0.0
        except:
            auc = 0.0

        # Sensitivity (Recall for positive class)
        if len(np.unique(all_labels)) > 1:
            _, recall_per_class, _, _ = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )
            sensitivity = (
                recall_per_class[1] * 100 if len(recall_per_class) > 1 else 0.0
            )
        else:
            sensitivity = 0.0

        # ✅ PRINT METRICS (same format as train)
        print("\n" + "=" * 50)
        print("Test Results:")
        print("=" * 50)
        print(f"Test Accuracy : {test_acc * 100:.2f}% | AUC: {auc:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}% | Sens: {sensitivity:.2f}%")
        print(f"Test F1-Score : {f1 * 100:.2f}%")
        print(f"Test IoU      : {test_iou * 100:.2f}% | mAP@0.5: {test_map * 100:.2f}%")
        print("\nClassification Report:")
        # FIXED: Ensure class_names are strings for sklearn
        class_names_str = [str(c) for c in class_names] if class_names else None
        print(
            classification_report(
                all_labels, all_preds, target_names=class_names_str, zero_division=0
            )
        )
        print("=" * 50)

        # Task 2: Save full model
        print("\n" + "=" * 50)
        print("Task 2: Save Full Model (DETR)")
        print("=" * 50)

        # Get gradcam layer
        model_name = model_type.lower()
        if isinstance(model, nn.DataParallel):
            gradcam_layer = get_gradcam_layer(model.module, model_name)
            model_to_save = model.module
        else:
            gradcam_layer = get_gradcam_layer(model, model_name)
            model_to_save = model

        # Calculate inference time
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

        # Save full model
        full_model_dir = os.path.join(output, "models")
        os.makedirs(full_model_dir, exist_ok=True)
        full_model_path = os.path.join(full_model_dir, f"{model_filename}_full.pth")

        try:
            torch.save(model_to_save, full_model_path)
            print(f"✅ Saved full DETR model to: {full_model_path}")
            print(f"   Model name: {model_type}")
            print(f"   Input size: {actual_input_size}")
            print(f"   Num queries: {num_queries}")
            print(f"   GradCAM layer: {gradcam_layer}")
            print(f"   Inference time: {inference_time:.4f}s")
            print(f"   Test Accuracy: {test_acc * 100:.2f}%")
            print(f"   Test IoU: {test_iou * 100:.2f}%")
            print(f"   Test mAP@0.5: {test_map * 100:.2f}%")
        except Exception as e:
            print(f"⚠️ Error saving full model: {e}")

    # Task 3: Visualize results with multiple queries
    if save_visualizations:
        print("\n" + "=" * 50)
        print("Task 3: Generate Visualizations (DETR Multiple Queries)")
        if use_gradcam:
            print("✓ GradCAM visualization enabled")
        print("=" * 50)

        vis_dir = os.path.join(output, "test_detr", model_filename)
        os.makedirs(vis_dir, exist_ok=True)

        # Create image_id index from test_df
        test_image_ids = test_df["image_id"].unique().tolist()
        print(f"Total test images: {len(test_image_ids)}")

        model.eval()
        batch_size_actual = batch_size

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                # Calculate image_ids
                start_idx = batch_idx * batch_size_actual
                end_idx = min(start_idx + len(images), len(test_image_ids))
                batch_image_ids = test_image_ids[start_idx:end_idx]

                # FIXED: Request attention maps explicitly
                outputs = model(images, return_attention_maps=True)
                cls_logits = outputs["cls_logits"]
                pred_bboxes = outputs["pred_bboxes"]
                pred_obj = torch.sigmoid(outputs["obj_scores"])
                attn_maps = outputs.get("attn_maps", None)

                _, predicted = torch.max(cls_logits, 1)
                probs = torch.softmax(cls_logits, dim=1)

                for i in range(len(images)):
                    if i >= len(batch_image_ids):
                        continue

                    image_id = str(batch_image_ids[i])
                    pred_class = predicted[i].item()
                    gt_label = labels[i].item()
                    pred_prob = probs[i, pred_class].item()

                    # Get image info
                    if image_id in image_info:
                        info = image_info[image_id]
                        original_size = info["original_size"]
                        image_path = info["image_path"]
                        gt_bbox_list = info.get("bbx_list", None)
                    else:
                        print(
                            f"⚠️ Warning: image_id '{image_id}' not found in image_info"
                        )
                        original_size = actual_input_size
                        image_path = None
                        gt_bbox_list = None

                    if original_size is None:
                        continue

                    # Get attention map - FIXED: should be [B, H, W]
                    if attn_maps is not None and len(attn_maps.shape) == 3:
                        attn_map = attn_maps[i]  # [H, W]
                    else:
                        print(
                            f"⚠️ Warning: No attention map or wrong shape for {image_id}"
                        )
                        if attn_maps is not None:
                            print(f"   attn_maps shape: {attn_maps.shape}")
                        continue

                    # Get all predicted bboxes and scores for this image
                    img_pred_bboxes = pred_bboxes[i]  # [N, 4]
                    img_pred_scores = pred_obj[i].squeeze(-1)  # [N]

                    # ADDED: Generate GradCAM if enabled
                    gradcam_map = None
                    if use_gradcam and gradcam_layer is not None:
                        try:
                            # Enable gradients temporarily
                            input_tensor = (
                                images[i : i + 1].clone().requires_grad_(True)
                            )

                            # Get prediction for this sample
                            with torch.set_grad_enabled(True):
                                if isinstance(model, nn.DataParallel):
                                    test_model = model.module
                                else:
                                    test_model = model

                                gradcam_map = gradcam(
                                    test_model,
                                    input_tensor,
                                    gradcam_layer,
                                    class_idx=pred_class,
                                )
                        except Exception as e:
                            print(f"⚠️ Warning: GradCAM failed for {image_id}: {e}")
                            gradcam_map = None

                    # Save visualization
                    save_path = os.path.join(vis_dir, f"{image_id}.png")
                    try:
                        visualize_m2_detr_result(
                            images[i],
                            attn_map,
                            img_pred_bboxes,
                            img_pred_scores,
                            gt_bbox_list,
                            pred_class,
                            gt_label,
                            pred_prob,
                            save_path,
                            class_names,
                            original_size,
                            image_path=image_path,
                            obj_threshold=obj_threshold,
                            use_otsu=use_otsu,
                            gradcam_map=gradcam_map,  # ADDED: Pass GradCAM
                        )
                    except Exception as e:
                        print(f"⚠️ Warning: Error visualizing {image_id}: {e}")
                        import traceback

                        traceback.print_exc()
                        continue

        print(f"✅ Saved DETR visualizations to: {vis_dir}")

    if not only_viz:
        return test_acc, test_iou
    return None, None


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
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--max_objects", type=int, default=3)
    parser.add_argument("--no_viz", action="store_true", help="Skip visualizations")
    parser.add_argument(
        "--only_viz",
        action="store_true",
        help="Only run visualization, skip evaluation and model saving",
    )
    parser.add_argument(
        "--sample_viz",
        action="store_true",
        help="Visualize test batches before running inference",
    )
    parser.add_argument(
        "--obj_threshold",
        type=float,
        default=0.5,
        help="Objectness threshold for displaying predictions",
    )
    parser.add_argument(
        "--use_otsu",
        action="store_true",
        help="Use Otsu thresholding for attention map visualization",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Add GradCAM visualization panel",
    )

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

    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    run_m2_detr_test_with_visualization(
        data_folder=data_folder,
        model_type=model_type,
        batch_size=batch_size,
        output=output,
        config_path=args.config,
        img_size=img_size,
        pretrained_model_path=pretrained_model_path,
        target_column=target_column,
        num_queries=args.num_queries,
        max_objects=args.max_objects,
        save_visualizations=not args.no_viz,
        only_viz=args.only_viz,
        sample_viz=args.sample_viz,
        obj_threshold=args.obj_threshold,
        use_otsu=args.use_otsu,
        use_gradcam=args.gradcam,  # ADDED: Pass argument
    )
