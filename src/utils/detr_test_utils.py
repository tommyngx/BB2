"""
Testing utilities for DETR models
- Independent evaluation functions
- Metrics computation
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

from src.utils.detr_common_utils import box_iou


def detr_compute_iou(pred_box, gt_box):
    """
    Compute IoU between single boxes - independent DETR implementation
    Args:
        pred_box: [4] tensor in normalized COCO format
        gt_box: [4] tensor in normalized COCO format
    Returns:
        iou: scalar
    """
    if pred_box.dim() == 1:
        pred_box = pred_box.unsqueeze(0)
    if gt_box.dim() == 1:
        gt_box = gt_box.unsqueeze(0)

    iou = box_iou(pred_box, gt_box)[0, 0]
    return iou


def evaluate_detr_model(model, test_loader, device):
    """Evaluate DETR model - independent implementation"""
    model.eval()
    correct, total = 0, 0
    total_iou = 0.0
    num_bbox_samples = 0

    all_preds, all_labels, all_probs = [], [], []
    all_pred_boxes, all_gt_boxes = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bboxes"].to(device)
            bbox_mask = batch["bbox_mask"].to(device)

            outputs = model(images)

            # Classification
            probs = torch.softmax(outputs["cls_logits"], dim=1)
            _, predicted = torch.max(outputs["cls_logits"], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Bbox evaluation
            pred_bboxes = outputs["pred_bboxes"]
            pred_obj = torch.sigmoid(outputs["obj_scores"])

            for i in range(images.size(0)):
                valid_mask = bbox_mask[i] > 0.5
                if valid_mask.sum() > 0:
                    top_idx = pred_obj[i].squeeze(-1).argmax()
                    pred_box = pred_bboxes[i, top_idx : top_idx + 1]
                    gt_boxes = bboxes[i][valid_mask]

                    iou = detr_compute_iou(pred_box[0], gt_boxes[0])
                    total_iou += iou.item()
                    num_bbox_samples += 1

                    all_pred_boxes.append(pred_box.cpu())
                    all_gt_boxes.append(gt_boxes.cpu())

    # Metrics
    test_acc = (correct / total) * 100
    test_iou = total_iou / max(num_bbox_samples, 1)

    # mAP@0.5
    test_map = 0.0
    if len(all_pred_boxes) > 0:
        num_correct = sum(
            1
            for pred_box, gt_box in zip(all_pred_boxes, all_gt_boxes)
            if detr_compute_iou(pred_box[0], gt_box[0]).item() >= 0.5
        )
        test_map = num_correct / len(all_pred_boxes)

    return {
        "accuracy": test_acc,
        "iou": test_iou,
        "map": test_map,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


def compute_classification_metrics(all_preds, all_labels, all_probs, class_names):
    """Compute detailed classification metrics"""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    try:
        auc = (
            roc_auc_score(all_labels, all_probs[:, 1]) * 100
            if all_probs.shape[1] == 2
            else 0.0
        )
    except:
        auc = 0.0

    if len(np.unique(all_labels)) > 1:
        _, recall_per_class, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        sensitivity = recall_per_class[1] * 100 if len(recall_per_class) > 1 else 0.0
    else:
        sensitivity = 0.0

    class_names_str = [str(c) for c in class_names] if class_names else None
    report = classification_report(
        all_labels, all_preds, target_names=class_names_str, zero_division=0
    )

    return {
        "accuracy": None,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "auc": auc,
        "sensitivity": sensitivity,
        "report": report,
    }


def print_test_metrics(metrics, test_iou, test_map):
    """Print concise test metrics"""
    # Calculate specificity if possible
    spec = None
    if metrics.get("report"):
        # Try to extract specificity from classification_report
        try:
            # Parse the report string for support/recall for class 0
            lines = metrics["report"].splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] == "0":
                    # specificity = recall of class 0
                    spec = float(parts[2]) * 100
                    break
        except Exception:
            spec = None

    acc = metrics.get("accuracy", 0.0)
    auc = metrics.get("auc", 0.0)
    sen = metrics.get("sensitivity", 0.0)
    f1 = metrics.get("f1", 0.0)
    iou = test_iou * 100
    map50 = test_map * 100

    print(
        f"Test Class: ACC={acc:.2f}%, AUC={auc:.2f}%, SEN={sen:.2f}%, SPEC={spec:.2f}%"
        if spec is not None
        else f"Test Class: ACC={acc:.2f}%, AUC={auc:.2f}%, SEN={sen:.2f}%"
    )
    print(f"Test Det: IoU={iou:.2f}%, mAP={map50:.2f}%, F1={f1:.2f}%")
