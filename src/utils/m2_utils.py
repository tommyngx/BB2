import torch
import math


def compute_iou(pred_boxes, gt_boxes):
    """
    Compute IoU between predicted and ground truth boxes (COCO format)
    Args:
        pred_boxes: [N, 4] or [B, N, 4] in format [x, y, w, h]
        gt_boxes: [M, 4] or [B, M, 4] in format [x, y, w, h]
    Returns:
        iou: [N, M] or [B, N, M]
    """
    # Get intersection coords
    x1_i = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1_i = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2_i = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2_i = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    # Intersection area
    inter_w = torch.clamp(x2_i - x1_i, min=0)
    inter_h = torch.clamp(y2_i - y1_i, min=0)
    inter_area = inter_w * inter_h

    # Union area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    )
    true_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = pred_area + true_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def get_lambda_bbox_schedule(
    epoch, lambda_bbox_target=0.5, freeze_epochs=10, warmup_epochs=10
):
    """
    Lambda bbox schedule with cosine warmup (smoother than linear)

    - Epoch 0-9: lambda = 0 (chỉ học classification)
    - Epoch 10-19: lambda tăng dần từ 0 lên lambda_bbox_target (cosine)
    - Epoch 20+: lambda = lambda_bbox_target
    """
    if epoch < freeze_epochs:
        return 0.0
    elif epoch < freeze_epochs + warmup_epochs:
        # Cosine warmup instead of linear
        progress = (epoch - freeze_epochs) / warmup_epochs
        cosine_progress = (1 - math.cos(progress * math.pi)) / 2
        return lambda_bbox_target * cosine_progress
    else:
        return lambda_bbox_target
