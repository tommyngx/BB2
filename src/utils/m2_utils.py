import torch
import math


def compute_iou(pred_boxes, gt_boxes):
    """
    Compute IoU between predicted and ground truth boxes (COCO format [x, y, w, h])
    Args:
        pred_boxes: [N, 4] in format [x, y, w, h] (normalized)
        gt_boxes: [M, 4] in format [x, y, w, h] (normalized)
    Returns:
        iou: [N, M] IoU matrix
    """
    # CRITICAL FIX: Ensure proper broadcasting by unsqueezing dimensions
    # pred_boxes: [N, 4] -> [N, 1, 4] for broadcasting
    # gt_boxes: [M, 4] -> [1, M, 4] for broadcasting

    pred = pred_boxes.unsqueeze(1)  # [N, 1, 4]
    gt = gt_boxes.unsqueeze(0)  # [1, M, 4]

    # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred[..., 0]  # [N, 1]
    pred_y1 = pred[..., 1]  # [N, 1]
    pred_x2 = pred[..., 0] + pred[..., 2]  # [N, 1]
    pred_y2 = pred[..., 1] + pred[..., 3]  # [N, 1]

    gt_x1 = gt[..., 0]  # [1, M]
    gt_y1 = gt[..., 1]  # [1, M]
    gt_x2 = gt[..., 0] + gt[..., 2]  # [1, M]
    gt_y2 = gt[..., 1] + gt[..., 3]  # [1, M]

    # Intersection coordinates [N, M]
    x1_i = torch.max(pred_x1, gt_x1)
    y1_i = torch.max(pred_y1, gt_y1)
    x2_i = torch.min(pred_x2, gt_x2)
    y2_i = torch.min(pred_y2, gt_y2)

    # Intersection area [N, M]
    inter_w = torch.clamp(x2_i - x1_i, min=0)
    inter_h = torch.clamp(y2_i - y1_i, min=0)
    inter_area = inter_w * inter_h

    # Box areas [N, 1] and [1, M]
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)  # [N, 1]
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # [1, M]

    # Union area [N, M]
    union_area = pred_area + gt_area - inter_area

    # IoU [N, M]
    iou = inter_area / (union_area + 1e-6)
    return iou


def get_lambda_bbox_schedule(
    epoch, lambda_bbox_target=0.5, freeze_epochs=10, warmup_epochs=10
):
    """Lambda bbox schedule with cosine warmup"""
    if epoch < freeze_epochs:
        return 0.0
    elif epoch < freeze_epochs + warmup_epochs:
        progress = (epoch - freeze_epochs) / warmup_epochs
        cosine_progress = (1 - math.cos(progress * math.pi)) / 2
        return lambda_bbox_target * cosine_progress
    else:
        return lambda_bbox_target
