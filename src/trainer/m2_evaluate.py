import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
)

from src.utils.m2_utils import compute_iou
from src.utils.bbox_loss import GIoULoss


def evaluate_m2_model(
    model,
    data_loader,
    device="cpu",
    mode="Test",
    return_loss=False,
    cls_criterion=None,
    bbox_criterion=None,
    lambda_bbox=1.0,
):
    """Evaluate multi-task model"""
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_iou = 0.0
    num_bbox_samples = 0

    if cls_criterion is None:
        cls_criterion = nn.CrossEntropyLoss()
    if bbox_criterion is None:
        bbox_criterion = GIoULoss()

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            cls_outputs, bbox_outputs = model(images)

            # Classification loss
            cls_loss = cls_criterion(cls_outputs, labels)

            # Bbox loss (only for positive samples)
            if has_bbox.sum() > 0:
                pos_mask = has_bbox.bool()
                bbox_loss = bbox_criterion(bbox_outputs[pos_mask], bboxes[pos_mask])
            else:
                bbox_loss = torch.tensor(0.0).to(device)

            # Total loss
            total_loss += (
                cls_loss.item() + lambda_bbox * bbox_loss.item()
            ) * images.size(0)

            # Compute IoU for positive samples
            if has_bbox.sum() > 0:
                pos_mask = has_bbox.bool()
                iou = compute_iou(bbox_outputs[pos_mask], bboxes[pos_mask])
                total_iou += iou.sum().item()
                num_bbox_samples += has_bbox.sum().item()

            # Classification metrics
            _, predicted = torch.max(cls_outputs, 1)
            probs = torch.softmax(cls_outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(
                probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy()
            )

    acc = correct / total
    avg_loss = total_loss / total
    avg_iou = total_iou / max(num_bbox_samples, 1)

    # Calculate metrics
    try:
        if len(set(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = None
    except Exception:
        auc = None

    try:
        precision = precision_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
        recall = recall_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        precision = 0.0
        recall = 0.0

    # Print results
    acc_loss_str = f"{mode} Accuracy : {acc * 100:.2f}% | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}"
    if auc is not None:
        acc_loss_str += f" | AUC: {auc * 100:.2f}%"
    print(acc_loss_str)
    print(f"{mode} Precision: {precision * 100:.2f}% | Sens: {recall * 100:.2f}%")

    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    if return_loss:
        return avg_loss, acc, avg_iou
    return acc
